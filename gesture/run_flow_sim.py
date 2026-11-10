"""Stage 1: キーボード模擬によるインタラクションフロー実行"""
import sys
import os

# gesture/ ディレクトリをパスに追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from interaction.signals import SignalBus
from interaction.sinks import ConsoleSink, JsonlSink
from interaction.state_machine import StateMachine, State
from interaction.keyboard_input import key_to_event


def print_help():
    print("=== 部品受け渡しフロー シミュレーション (ステップモード) ===")
    print()
    print("キー操作:")
    print("  w  = WANT (ください)  confirm")
    print("  W  = WANT             onset")
    print("  y  = YES  (はい)      confirm")
    print("  Y  = YES              onset")
    print("  n  = NO   (いいえ)    confirm")
    print("  N  = NO               onset")
    print("  x  = CANCEL (戻る)    confirm")
    print("  1-5= POINT (P1〜P5)   confirm")
    print()
    print("  (空Enter / スペース) = 次へ進む (模擬動作完了)")
    print("  q  = 終了")
    print("  ?  = ヘルプ表示")
    print()


def main():
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")

    bus = SignalBus()
    console = ConsoleSink()
    jsonl = JsonlSink(os.path.join(os.path.dirname(__file__), "logs"))
    bus.subscribe(console)
    bus.subscribe(jsonl)

    sm = StateMachine(config_path, bus, step_mode=True)

    print_help()
    print(f"状態: {sm.state.name}  |  部品: {sm.parts.current}")
    print()

    try:
        while True:
            # 保留イベントがあればプロンプトに表示
            pending_hint = ""
            if sm.pending:
                pending_hint = f" [スペース→{sm.pending.kind.name}]"

            raw = input(f"[{sm.state.name}]{pending_hint} > ").strip()

            # 空入力 or スペースのみ → step (保留イベント実行)
            if raw == "" or raw.isspace():
                if sm.pending:
                    print(f"  → {sm.pending}")
                    sm.step()
                    print(f"  状態: {sm.state.name}  |  部品: {sm.parts.current}")
                    print()
                else:
                    print("  (保留中のイベントなし)")
                continue

            if raw == "q":
                break
            if raw == "?":
                print_help()
                continue

            for ch in raw:
                if ch == " ":
                    if sm.pending:
                        print(f"  → {sm.pending}")
                        sm.step()
                    continue
                ev = key_to_event(ch)
                if ev is None:
                    print(f"  (不明なキー: {ch})")
                    continue
                print(f"  → {ev}")
                sm.handle(ev)

            print(f"  状態: {sm.state.name}  |  部品: {sm.parts.current}")
            print()

    except (KeyboardInterrupt, EOFError):
        print("\n終了")
    finally:
        jsonl.close()


if __name__ == "__main__":
    main()
