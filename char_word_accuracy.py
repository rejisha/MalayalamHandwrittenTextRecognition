def calculate_accuracy(reference, predicted):
    '''
    Calculate character-level and word-level accuracy between reference and predicted sentences.
    '''

    # Character-level accuracy
    ref_chars = list(reference)
    print('ref ', ref_chars)
    pred_chars = list(predicted)
    print('pred_chars ', pred_chars)

    correct_chars = sum(1 for ref, pred in zip(ref_chars, pred_chars) if ref == pred)
    total_chars = len(reference)
    char_accuracy = correct_chars / total_chars if total_chars > 0 else 0

    # Word-level accuracy
    ref_words = reference.split()
    pred_words = predicted.split()

    correct_words = sum(1 for ref, pred in zip(ref_words, pred_words) if ref == pred)
    total_words = len(ref_words)
    word_accuracy = correct_words / total_words if total_words > 0 else 0

    return char_accuracy * 100, word_accuracy * 100


# Example usage
if __name__ == "__main__":
    reference_sentence = "ഇടിയും മഴയുമായി  കുട്ടികൾ അലറുന്നതു കേട്ടു അമ്മ തത്ത പറന്നു കൂട്ടിൽ കയറി"
    predicted_sentence = "ഇടിയും മഴയുമായി കൂടിയായാൽ അലിഞ്ഞു കേടു അമ്മ തത്ത പണൻ കൂടിൽ കയറി"

    char_acc, word_acc = calculate_accuracy(reference_sentence, predicted_sentence)
    print(f"Character-level accuracy: {char_acc:.2f}%")
    print(f"Word-level accuracy: {word_acc:.2f}%")
