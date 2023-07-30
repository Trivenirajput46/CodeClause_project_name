import math#importing lib

#creating functions
def add(x, y):#for addition
    return x + y

def subtract(x, y):#for subtraction
    return x - y

def multiply(x, y):
    return x * y

def divide(x, y):
    if y == 0:
        return "Cannot divide by zero!"
    return x / y

def square_root(x):
    if x < 0:
        return "Invalid input! Cannot calculate the square root of a negative number."
    return math.sqrt(x)

def exponentiation(x, y):
    return x ** y

def sin_degrees(x):
    return math.sin(math.radians(x))

def cos_degrees(x):
    return math.cos(math.radians(x))

def tan_degrees(x):
    return math.tan(math.radians(x))

def main():
    print("Welcome to the Advanced Calculator!")
    
    #loop
    while True:
        print("\nAvailable operations:")
        print("1. Add")
        print("2. Subtract")
        print("3. Multiply")
        print("4. Divide")
        print("5. Square Root")
        print("6. Exponentiation")
        print("7. Sine (in degrees)")
        print("8. Cosine (in degrees)")
        print("9. Tangent (in degrees)")
        print("0. Exit")
        
        choice = input("\nEnter your choice (0-9): ")
        
        if choice == '0':
            print("Thank you for using the calculator. Goodbye!")
            break
        
        if choice not in ['1', '2', '3', '4', '5', '6', '7', '8', '9']:
            print("Invalid choice! Please select a valid option.")
            continue
        
        if choice in ['5', '7', '8', '9']:
            num1 = float(input("Enter the number: "))
        else:
            num1 = float(input("Enter the first number: "))
            num2 = float(input("Enter the second number: "))
        
        if choice == '1':
            print("Result:", add(num1, num2))
        elif choice == '2':
            print("Result:", subtract(num1, num2))
        elif choice == '3':
            print("Result:", multiply(num1, num2))
        elif choice == '4':
            print("Result:", divide(num1, num2))
        elif choice == '5':
            print("Result:", square_root(num1))
        elif choice == '6':
            print("Result:", exponentiation(num1, num2))
        elif choice == '7':
            print("Result:", sin_degrees(num1))
        elif choice == '8':
            print("Result:", cos_degrees(num1))
        elif choice == '9':
            print("Result:", tan_degrees(num1))
        
if __name__ == "__main__":
    main()
