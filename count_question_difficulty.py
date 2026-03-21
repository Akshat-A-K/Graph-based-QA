import json

def count_difficulty_questions(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    easy = medium = hard = 0
    for item in data:
        difficulty = item.get('level', '').lower()
        if difficulty == 'easy':
            easy += 1
        elif difficulty == 'medium':
            medium += 1
        elif difficulty == 'hard':
            hard += 1
    return easy, medium, hard

if __name__ == '__main__':
    # Change the file name as needed
    json_file = 'hotpot_train_v1.1.json'
    easy, medium, hard = count_difficulty_questions(json_file)
    print(f"easy: {easy}")
    print(f"medium: {medium}")
    print(f"hard: {hard}")
