import pytest

def main():  # import actual main function
    return "Hello, world!"


def test_output():
    expected_output = "Hello, world!"  # replace with the expected output of your main function
    actual_output = main()  # call your main function and get its output
    assert actual_output == expected_output, "Main function output does not match expected output"
