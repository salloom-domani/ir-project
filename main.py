from services.clean_text import clean_and_tokenize


def main():
    print("Hello from ir-project!")

    sample = "This is a sample text! Visit: https://example.com123"
    result = clean_and_tokenize(sample)
    print(result)


if __name__ == "__main__":
    main()
