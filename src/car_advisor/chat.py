from car_advisor.graph import run_turn


def main():
    session_id = input("session id> ").strip()
    if not session_id:
        raise SystemExit("a session id is required")
    while True:
        user_text = input("you> ").strip()
        if user_text == "quit":
            break
        if not user_text:
            continue
        final = run_turn(session_id, user_text)
        if final is None:
            print("advisor> (no answer)")
        else:
            print("advisor>", final.answer)


if __name__ == "__main__":
    main()
