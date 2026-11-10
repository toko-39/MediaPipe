"""状態機械のテスト — 9 シナリオ"""
import sys
import os
import time
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from interaction.events import Event, EventKind, Stage
from interaction.signals import SignalBus, SignalKind
from interaction.state_machine import StateMachine, State


@pytest.fixture
def setup():
    """StateMachine + シグナル記録"""
    bus = SignalBus()
    log = []
    bus.subscribe(lambda sig: log.append(sig.kind))
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    sm = StateMachine(config_path, bus)
    return sm, log


def confirm(kind):
    return Event(kind, Stage.CONFIRM)


def onset(kind):
    return Event(kind, Stage.ONSET)


def wait_sim(sec=0.5):
    """模擬動作の完了を待つ (テスト用に短め設定のため余裕を持って待つ)"""
    time.sleep(sec)


# ── シナリオ 1: 正常フロー (WANT → YES → 受け渡し完了) ──

class TestNormalFlow:
    def test_want_yes_handover(self, setup):
        sm, log = setup
        assert sm.state == State.IDLE

        sm.handle(confirm(EventKind.WANT))
        assert sm.state == State.ACCEPTING
        assert SignalKind.ACCEPT in log

        # ACTION_DONE を待つ (accept_duration=2.0 だがテストでは待つ)
        time.sleep(2.5)
        assert sm.state == State.WAIT_REACTION
        assert SignalKind.OFFER_PART in log

        sm.handle(confirm(EventKind.YES))
        assert sm.state == State.ACKING
        assert SignalKind.ACK in log

        time.sleep(2.5)
        assert sm.state == State.HANDING_OVER
        assert SignalKind.HANDOVER in log

        time.sleep(3.5)
        assert sm.state == State.IDLE


# ── シナリオ 2: onset → confirm 2段階 ──

class TestTwoStageDetection:
    def test_onset_then_confirm(self, setup):
        sm, log = setup

        sm.handle(onset(EventKind.WANT))
        assert sm.state == State.REQ_DETECTING
        assert SignalKind.LISTENING in log

        sm.handle(confirm(EventKind.WANT))
        assert sm.state == State.ACCEPTING


# ── シナリオ 3: REQ_DETECTING タイムアウト ──

class TestReqTimeout:
    def test_timeout_returns_idle(self, setup):
        sm, log = setup

        sm.handle(onset(EventKind.WANT))
        assert sm.state == State.REQ_DETECTING

        # req_detecting タイムアウト = 3.0s
        time.sleep(3.5)
        assert sm.state == State.IDLE
        assert SignalKind.FLOW_RESET in log


# ── シナリオ 4: ACCEPTING 中の CANCEL ──

class TestAcceptingCancel:
    def test_cancel_during_accept(self, setup):
        sm, log = setup

        sm.handle(confirm(EventKind.WANT))
        assert sm.state == State.ACCEPTING

        sm.handle(confirm(EventKind.CANCEL))
        assert sm.state == State.IDLE
        assert SignalKind.FLOW_RESET in log


# ── シナリオ 5: NO → POINT → CONFIRM ──

class TestNoPointConfirm:
    def test_no_then_point(self, setup):
        sm, log = setup

        sm.handle(confirm(EventKind.WANT))
        time.sleep(2.5)  # ACCEPTING → WAIT_REACTION

        sm.handle(confirm(EventKind.NO))
        assert sm.state == State.WAIT_POINTING

        sm.handle(Event(EventKind.POINT, Stage.CONFIRM, "P2"))
        assert sm.state == State.CONFIRMING
        assert SignalKind.CONFIRM_PART in log

        sm.handle(confirm(EventKind.YES))
        assert sm.state == State.ACKING


# ── シナリオ 6: WAIT_REACTION タイムアウト → ATTENTION ──

class TestAttention:
    def test_attention_then_reset(self, setup):
        sm, log = setup

        sm.handle(confirm(EventKind.WANT))
        time.sleep(2.5)  # → WAIT_REACTION
        assert sm.state == State.WAIT_REACTION

        # wait_reaction=5.0, attention_max=3
        # 1回目
        time.sleep(5.5)
        assert SignalKind.ATTENTION in log
        assert sm.state == State.WAIT_REACTION

        # 2回目
        time.sleep(5.5)
        count = log.count(SignalKind.ATTENTION)
        assert count >= 2

        # 3回目 → IDLE
        time.sleep(5.5)
        assert sm.state == State.IDLE


# ── シナリオ 7: ACKING 中の CANCEL ──

class TestAckingCancel:
    def test_cancel_during_ack(self, setup):
        sm, log = setup

        sm.handle(confirm(EventKind.WANT))
        time.sleep(2.5)  # → WAIT_REACTION

        sm.handle(confirm(EventKind.YES))
        assert sm.state == State.ACKING

        sm.handle(confirm(EventKind.CANCEL))
        assert sm.state == State.WAIT_REACTION
        # OFFER_PART が再度出る
        assert log.count(SignalKind.OFFER_PART) >= 2


# ── シナリオ 8: REACTION_DETECTING タイムアウト ──

class TestReactionDetectingTimeout:
    def test_onset_timeout_returns_wait(self, setup):
        sm, log = setup

        sm.handle(confirm(EventKind.WANT))
        time.sleep(2.5)  # → WAIT_REACTION

        sm.handle(onset(EventKind.YES))
        assert sm.state == State.REACTION_DETECTING

        # reaction_detecting=3.0
        time.sleep(3.5)
        assert sm.state == State.WAIT_REACTION


# ── シナリオ 9: 全部品完了 ──

class TestFlowComplete:
    def test_all_parts_complete(self, setup):
        sm, log = setup

        # 5部品を順に完了
        for i in range(5):
            sm.handle(confirm(EventKind.WANT))
            time.sleep(2.5)  # → WAIT_REACTION
            sm.handle(confirm(EventKind.YES))
            time.sleep(2.5)  # → HANDING_OVER
            time.sleep(3.5)  # → IDLE

        assert sm.parts.is_complete
        assert SignalKind.FLOW_COMPLETE in log
