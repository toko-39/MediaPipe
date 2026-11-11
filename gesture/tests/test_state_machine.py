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
# Stage 2 で変更: ACCEPTING 中の CANCEL は無視される (遷移テーブルにエントリなし)

class TestAcceptingCancel:
    def test_cancel_during_accept_is_ignored(self, setup):
        sm, log = setup

        sm.handle(confirm(EventKind.WANT))
        assert sm.state == State.ACCEPTING

        sm.handle(confirm(EventKind.CANCEL))
        # Stage 2: ACCEPTING 中の CANCEL は無視され、状態は変わらない
        assert sm.state == State.ACCEPTING


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
# Stage 2 で変更: ACKING 中の CANCEL → WAIT_POINTING に遷移 (WAIT_REACTION ではない)

class TestAckingCancel:
    def test_cancel_during_ack(self, setup):
        sm, log = setup

        sm.handle(confirm(EventKind.WANT))
        time.sleep(2.5)  # → WAIT_REACTION

        sm.handle(confirm(EventKind.YES))
        assert sm.state == State.ACKING

        sm.handle(confirm(EventKind.CANCEL))
        # Stage 2: WAIT_POINTING に遷移 (NOをYESと誤解釈した場合の訂正)
        assert sm.state == State.WAIT_POINTING


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


# ── シナリオ 9: 複数回の受け渡し ──
# Stage 2 で変更: FLOW_COMPLETE 削除。完了カウントなし、毎回ランダム選択で IDLE に戻る

class TestMultipleHandovers:
    def test_multiple_handovers_return_to_idle(self, setup):
        sm, log = setup

        # 3回受け渡しを実行し、毎回 IDLE に戻ることを確認
        for i in range(3):
            sm.handle(confirm(EventKind.WANT))
            time.sleep(2.5)  # → WAIT_REACTION
            sm.handle(confirm(EventKind.YES))
            time.sleep(2.5)  # → HANDING_OVER
            time.sleep(3.5)  # → IDLE
            assert sm.state == State.IDLE

        # FLOW_RESET が毎回発行される (FLOW_COMPLETE は存在しない)
        assert log.count(SignalKind.FLOW_RESET) >= 3
