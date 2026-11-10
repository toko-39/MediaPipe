"""状態機械 — データ駆動遷移テーブル"""
from __future__ import annotations
import threading
import time
from enum import Enum, auto
from typing import Optional

import yaml

from interaction.events import Event, EventKind, Stage
from interaction.signals import Signal, SignalBus, SignalKind
from interaction.parts import PartsManager
from interaction.simulated import SimulatedCompletion


class State(Enum):
    IDLE = auto()
    REQ_DETECTING = auto()
    ACCEPTING = auto()
    WAIT_REACTION = auto()
    REACTION_DETECTING = auto()
    WAIT_POINTING = auto()
    CONFIRMING = auto()
    ACKING = auto()
    HANDING_OVER = auto()


class StateMachine:
    def __init__(self, config_path: str, signal_bus: SignalBus,
                 step_mode: bool = False):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        self._timeouts = cfg["timeouts"]
        self._sim_cfg = cfg["simulated"]
        self._bus = signal_bus
        self._parts = PartsManager(cfg["parts"])
        self._step_mode = step_mode

        self._state = State.IDLE
        self._attention_count = 0
        self._lock = threading.Lock()

        # タイムアウト用タイマー
        self._timeout_timer: Optional[threading.Timer] = None
        # 模擬動作完了
        self._sim_accept = SimulatedCompletion()
        self._sim_ack = SimulatedCompletion()
        self._sim_handover = SimulatedCompletion()

        # step_mode: 保留中のイベント (スペースキーで発火)
        self._pending_event: Optional[Event] = None

        # 遷移テーブル: (state, event_kind, stage) -> handler method
        self._table = {
            # --- IDLE ---
            (State.IDLE, EventKind.WANT, Stage.ONSET): self._idle_want_onset,
            (State.IDLE, EventKind.WANT, Stage.CONFIRM): self._idle_want_confirm,

            # --- REQ_DETECTING ---
            (State.REQ_DETECTING, EventKind.WANT, Stage.CONFIRM): self._req_want_confirm,
            (State.REQ_DETECTING, EventKind.TIMEOUT, Stage.CONFIRM): self._req_timeout,

            # --- ACCEPTING ---
            (State.ACCEPTING, EventKind.ACTION_DONE, Stage.CONFIRM): self._accepting_done,

            # --- WAIT_REACTION ---
            (State.WAIT_REACTION, EventKind.YES, Stage.ONSET): self._wait_yes_onset,
            (State.WAIT_REACTION, EventKind.YES, Stage.CONFIRM): self._wait_yes_confirm,
            (State.WAIT_REACTION, EventKind.NO, Stage.ONSET): self._wait_no_onset,
            (State.WAIT_REACTION, EventKind.NO, Stage.CONFIRM): self._wait_no_confirm,
            (State.WAIT_REACTION, EventKind.CANCEL, Stage.CONFIRM): self._cancel_to_idle,
            (State.WAIT_REACTION, EventKind.TIMEOUT, Stage.CONFIRM): self._wait_timeout,

            # --- REACTION_DETECTING ---
            (State.REACTION_DETECTING, EventKind.YES, Stage.CONFIRM): self._react_yes,
            (State.REACTION_DETECTING, EventKind.NO, Stage.CONFIRM): self._react_no,
            (State.REACTION_DETECTING, EventKind.TIMEOUT, Stage.CONFIRM): self._react_timeout,

            # --- ACKING ---
            (State.ACKING, EventKind.ACTION_DONE, Stage.CONFIRM): self._acking_done,
            (State.ACKING, EventKind.CANCEL, Stage.CONFIRM): self._cancel_to_wait_pointing,

            # --- HANDING_OVER ---
            (State.HANDING_OVER, EventKind.HANDOVER_DONE, Stage.CONFIRM): self._handover_done,
            (State.HANDING_OVER, EventKind.CANCEL, Stage.CONFIRM): self._cancel_to_wait_pointing,

            # --- WAIT_POINTING ---
            (State.WAIT_POINTING, EventKind.POINT, Stage.CONFIRM): self._pointing_received,
            (State.WAIT_POINTING, EventKind.TIMEOUT, Stage.CONFIRM): self._pointing_timeout,

            # --- CONFIRMING ---
            (State.CONFIRMING, EventKind.YES, Stage.CONFIRM): self._confirming_yes,
            (State.CONFIRMING, EventKind.NO, Stage.CONFIRM): self._confirming_no,
            (State.CONFIRMING, EventKind.CANCEL, Stage.CONFIRM): self._confirming_cancel,
            (State.CONFIRMING, EventKind.TIMEOUT, Stage.CONFIRM): self._confirming_timeout,
        }

    # ── public ──────────────────────────────────────

    @property
    def state(self) -> State:
        return self._state

    @property
    def parts(self) -> PartsManager:
        return self._parts

    @property
    def pending(self) -> Optional[Event]:
        """step_mode 時に保留中のイベント"""
        return self._pending_event

    def step(self):
        """step_mode: 保留中のイベントを実行"""
        if self._pending_event:
            ev = self._pending_event
            self._pending_event = None
            self.handle(ev)

    def handle(self, event: Event):
        with self._lock:
            key = (self._state, event.kind, event.stage)
            handler = self._table.get(key)
            if handler:
                handler(event)

    # ── helpers ─────────────────────────────────────

    def _set_state(self, new_state: State):
        self._state = new_state

    def _emit(self, kind: SignalKind, payload=None):
        self._bus.emit(Signal(kind, payload))

    def _start_timeout(self, name: str):
        self._cancel_timeout()
        if self._step_mode:
            return  # step_mode ではタイムアウト無効
        sec = self._timeouts[name]
        t = threading.Timer(sec, lambda: self.handle(
            Event(EventKind.TIMEOUT, Stage.CONFIRM, name)))
        t.daemon = True
        t.start()
        self._timeout_timer = t

    def _cancel_timeout(self):
        if self._timeout_timer:
            self._timeout_timer.cancel()
            self._timeout_timer = None

    def _start_sim(self, sim: SimulatedCompletion, duration: float,
                   event_kind: EventKind):
        """模擬タイマーを開始。step_mode では保留イベントとして登録"""
        if self._step_mode:
            self._pending_event = Event(event_kind, Stage.CONFIRM)
        else:
            sim.start(duration, event_kind, self.handle)

    # ── IDLE ────────────────────────────────────────

    def _idle_want_onset(self, ev: Event):
        self._emit(SignalKind.LISTENING)
        self._set_state(State.REQ_DETECTING)
        self._start_timeout("req_detecting")

    def _idle_want_confirm(self, ev: Event):
        # onset なしで直接 confirm が来た場合
        self._emit(SignalKind.LISTENING)
        self._do_accept()

    # ── REQ_DETECTING ──────────────────────────────

    def _req_want_confirm(self, ev: Event):
        self._cancel_timeout()
        self._do_accept()

    def _req_timeout(self, ev: Event):
        self._set_state(State.IDLE)
        self._emit(SignalKind.FLOW_RESET)

    # ── shared: start accept ───────────────────────

    def _do_accept(self):
        self._set_state(State.ACCEPTING)
        self._emit(SignalKind.ACCEPT)
        self._start_sim(self._sim_accept, self._sim_cfg["accept_duration"],
                        EventKind.ACTION_DONE)

    # ── ACCEPTING ──────────────────────────────────

    def _accepting_done(self, ev: Event):
        self._set_state(State.WAIT_REACTION)
        self._attention_count = 0
        self._emit(SignalKind.OFFER_PART, self._parts.current)
        self._start_timeout("wait_reaction")

    def _cancel_to_idle(self, ev: Event):
        self._cancel_timeout()
        self._set_state(State.IDLE)
        self._emit(SignalKind.FLOW_RESET)

    def _cancel_to_wait_pointing(self, ev: Event):
        self._cancel_timeout()
        self._sim_ack.cancel()
        self._sim_handover.cancel()
        self._pending_event = None
        self._do_wait_pointing()

    # ── WAIT_REACTION ──────────────────────────────

    def _wait_yes_onset(self, ev: Event):
        self._cancel_timeout()
        self._emit(SignalKind.WATCHING)
        self._set_state(State.REACTION_DETECTING)
        self._start_timeout("reaction_detecting")

    def _wait_yes_confirm(self, ev: Event):
        # onset なしで直接 confirm
        self._cancel_timeout()
        self._do_ack()

    def _wait_no_onset(self, ev: Event):
        self._cancel_timeout()
        self._emit(SignalKind.WATCHING)
        self._set_state(State.REACTION_DETECTING)
        self._start_timeout("reaction_detecting")

    def _wait_no_confirm(self, ev: Event):
        self._cancel_timeout()
        self._do_wait_pointing()

    def _wait_timeout(self, ev: Event):
        self._attention_count += 1
        if self._attention_count >= self._timeouts["attention_max"]:
            self._set_state(State.IDLE)
            self._emit(SignalKind.FLOW_RESET)
        else:
            self._emit(SignalKind.ATTENTION)
            self._start_timeout("wait_reaction")

    # ── REACTION_DETECTING ─────────────────────────

    def _react_yes(self, ev: Event):
        self._cancel_timeout()
        self._do_ack()

    def _react_no(self, ev: Event):
        self._cancel_timeout()
        self._do_wait_pointing()

    def _react_timeout(self, ev: Event):
        # confirm が来なかった → WAIT_REACTION に戻る
        self._set_state(State.WAIT_REACTION)
        self._start_timeout("wait_reaction")

    # ── shared: ack / wait_pointing ────────────────

    def _do_ack(self):
        self._set_state(State.ACKING)
        self._emit(SignalKind.ACK)
        self._start_sim(self._sim_ack, self._sim_cfg["ack_duration"],
                        EventKind.ACTION_DONE)

    def _do_wait_pointing(self):
        self._set_state(State.WAIT_POINTING)
        self._start_timeout("wait_pointing")

    # ── ACKING ─────────────────────────────────────

    def _acking_done(self, ev: Event):
        self._set_state(State.HANDING_OVER)
        self._emit(SignalKind.HANDOVER)
        self._start_sim(self._sim_handover, self._sim_cfg["handover_duration"],
                        EventKind.HANDOVER_DONE)

    # ── HANDING_OVER ───────────────────────────────

    def _handover_done(self, ev: Event):
        self._parts.pick_random()
        self._set_state(State.IDLE)
        self._emit(SignalKind.FLOW_RESET)

    # ── WAIT_POINTING ──────────────────────────────

    def _pointing_received(self, ev: Event):
        self._cancel_timeout()
        part_name = ev.payload
        if part_name and self._parts.select(part_name):
            self._set_state(State.CONFIRMING)
            self._emit(SignalKind.CONFIRM_PART, part_name)
            self._start_timeout("confirming")
        else:
            # 不正な部品 → タイムアウト再開
            self._start_timeout("wait_pointing")

    def _pointing_timeout(self, ev: Event):
        self._set_state(State.IDLE)
        self._emit(SignalKind.FLOW_RESET)

    # ── CONFIRMING ─────────────────────────────────

    def _confirming_yes(self, ev: Event):
        self._cancel_timeout()
        self._do_ack()

    def _confirming_no(self, ev: Event):
        self._cancel_timeout()
        self._do_wait_pointing()

    def _confirming_cancel(self, ev: Event):
        self._cancel_timeout()
        self._do_wait_pointing()

    def _confirming_timeout(self, ev: Event):
        self._set_state(State.IDLE)
        self._emit(SignalKind.FLOW_RESET)
