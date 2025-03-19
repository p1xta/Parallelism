import math


def parse_line(line):
    parts = line.split('=')
    task_id = int(parts[0].split()[2])
    operands = parts[1].strip()[1:-1]
    if ',' in operands:
        op1, op2 = map(float, operands.split(','))
    else:
        op1 = float(operands)
        op2 = None
    result = float(parts[2].strip())
    return task_id, op1, op2, result

def parse_file(filename):
    with open(filename, 'r') as file:
        operation = file.readline().strip()
        results = []
        for line in file:
            if not line.strip():
                continue
            results.append(parse_line(line))
        return operation, results

def perform_operation(operation, op1, op2):
    if operation == 'sin':
        return math.sin(op1)
    if operation == 'sqrt':
        return math.sqrt(op1)
    elif operation == 'pow':
        return math.pow(op1, op2)
    else:
        raise ValueError(f"Unknown operation: {operation}")

def check_results(filename):
    operation, results = parse_file(filename)
    print(f"\nFile {filename}:")
    for task_id, op1, op2, expected_result in results:
        actual_result = perform_operation(operation, op1, op2)
        if math.isclose(actual_result, expected_result, rel_tol=1e-3):
            print(f"Task id {task_id} - CORRECT")
        else:
            print(f"Task id {task_id} - WRONG (expected {expected_result}, got {actual_result})")

check_results('client1.txt')
check_results('client2.txt')
check_results('client3.txt')