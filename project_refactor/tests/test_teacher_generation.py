from unlearning_research.teacher import replace_person_name, replacement_schedule


def test_replacement_schedule_is_nested():
    pool = ["A Person", "B Person", "C Person", "D Person"]
    first_two = replacement_schedule(pool, target_name="Target Person", num_samples=2, seed=7)
    first_four = replacement_schedule(pool, target_name="Target Person", num_samples=4, seed=7)
    assert first_four[:2] == first_two


def test_replace_person_name_handles_full_and_last_name_mentions():
    text = "Paul Marston was a historian. Marston worked in Cambridge."
    out = replace_person_name(text, "Paul Marston", "Wilhelm Wattenbach")
    assert "Paul Marston" not in out
    assert "Marston" not in out
    assert "Wilhelm Wattenbach" in out
    assert "Wattenbach worked" in out
