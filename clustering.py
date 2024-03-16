""" DOCSTRING

  This program generates a list of unique students with certain properties,
  then attempts to perform K-means clustering on the students to seperate
  them into the desired number of groups


  TODO:

    We need to properly define the feature vector based on all student properties we want to compare.
    To do this, we will need to figure out how to convert the textual student properties (major) into a numerical value
    as well as how we will cluster based on the courselist of each student.

    Once these issues are figured out, we can decide on the best way to display the groups and maybe work on some sort of UI?
"""

import numpy as np

# CONSTANTS
num_students = 100



# Courses can be randomly generated usang a faculty code (ex: MECH),
#  followed by the course level(ex: 100) + a randomly chosen course number (ex: 41)
# In this case we would get (MECH141)
# The course_departments can also be used to represent the students major,
#  if needed we can define a dict that matches the faculty codes to the full major spelling (ex: PHYS -> Physics)
course_departments = ["CSC", "ECE", "MECH", "PHYS", "BIOL", "MATH", "PSYCH", "ENGL", "HIST", "CHEM"]
course_numbers = [0, 1, 10, 15, 20, 25, 30, 35, 40, 41, 50, 55, 60, 71, 86, 99]

# Set of first names sorted alphabetically
# A set in python only contains unique elements (we don't want duplicate names)
# Set must be converted back into a list to work with np.random.choice()
names = list(set([
    "Aaron", "Adam", "Adrian", "Adrienne", "Alan", "Albert", "Alberto", "Alexander", "Alfred",
    "Alice", "Alison", "Allen", "Allison", "Alma", "Alvin", "Amanda", "Amber", "Amy",
    "Andre", "Andrea", "Andrew", "Angela", "Angie", "Ann", "Anna", "Anthony", "Arlene",
    "Arthur", "Ashley", "Austin", "Barbara", "Beatrice", "Benjamin", "Betty", "Beverly", "Billy",
    "Bobbie", "Bobby", "Brandon", "Brenda", "Brett", "Brian", "Brittany", "Bruce", "Bryan",
    "Caleb", "Calvin", "Carl", "Carla", "Carol", "Caroline", "Carolyn", "Carrie", "Cassandra",
    "Catherine", "Charles", "Charlie", "Cheryl", "Chris", "Christian", "Christina", "Christine", "Christopher",
    "Christy", "Clarence", "Claudia", "Connie", "Corey", "Cory", "Crystal", "Curtis", "Cynthia",
    "Dale", "Damon", "Dan", "Daniel", "Danielle", "Dante", "Daryl", "David", "Dawn",
    "Deanna", "Deborah", "Debra", "Denise", "Dennis", "Derrick", "Diana", "Diane", "Dominic",
    "Don", "Donald", "Donna", "Doris", "Dorothy", "Douglas", "Dwayne", "Dwight", "Dylan",
    "Edith", "Edna", "Edward", "Edythe", "Eileen", "Elias", "Elijah", "Elizabeth", "Emily",
    "Emma", "Emmanuel", "Eric", "Erin", "Esther", "Ethan", "Eugene", "Evelyn", "Felicia",
    "Fernando", "Frances", "Frank", "Fred", "Gabriel", "Gabrielle", "Garrett", "Gary", "Gayle",
    "Gene", "George", "Gerald", "Gerardo", "Gina", "Gladys", "Glenn", "Gloria", "Gordon",
    "Grace", "Graham", "Gregg", "Gregory", "Gretchen", "Gwendolyn", "Hannah", "Harold", "Harry",
    "Harvey", "Hazel", "Heather", "Helen", "Henry", "Holly", "Ida", "Irma", "Isaac",
    "Isaiah", "Ivan", "Jack", "Jackie", "Jacob", "Jacqueline", "Jacquelyn", "Jake", "James",
    "Jane", "Janet", "Janice", "Jared", "Jason", "Jean", "Jeanette", "Jeffery", "Jeffrey",
    "Jennie", "Jennifer", "Jenny", "Jeremy", "Jerry", "Jesse", "Jessica", "Jesus", "Jill",
    "Joan", "Joanne", "Joe", "Joel", "John", "Jonathan", "Jordan", "Jose", "Joseph",
    "Joshua", "Joyce", "Juan", "Judith", "Judy", "Julia", "Julie", "Justin", "Karen",
    "Katherine", "Kathleen", "Kathryn", "Kathy", "Katie", "Keith", "Kelly", "Kelvin", "Kenneth",
    "Kerry", "Kevin", "Kim", "Kimberly", "Kirk", "Kyle", "Lane", "Larry", "Laura",
    "Lauren", "Lawrence", "Leon", "Leroy", "Leticia", "Lewis", "Linda", "Lisa", "Logan",
    "Lori", "Lorraine", "Louis", "Lucille", "Luis", "Lyle", "Lynette", "Lynn", "Madison",
    "Manuel", "Margaret", "Margie", "Maria", "Marie", "Marilyn", "Marion", "Mark", "Martha",
    "Mary", "Matthew", "Max", "Megan", "Melissa", "Michael", "Michele", "Michelle", "Mildred",
    "Milton", "Monica", "Monique", "Nancy", "Natasha", "Nathan", "Nathaniel", "Nicholas", "Nicole",
    "Nina", "Noah", "Nora", "Olivia", "Pamela", "Patrice", "Patricia", "Patrick", "Patsy",
    "Paul", "Paula", "Peggy", "Perry", "Peter", "Philip", "Preston", "Rachel", "Ralph",
    "Ramon", "Randy", "Raymond", "Rebecca", "Reginald", "Renee", "Ricardo", "Richard", "Robert",
    "Roberto", "Robin", "Rodney", "Roger", "Roland", "Roman", "Ronald", "Rosa", "Rose",
    "Roy", "Rudolph", "Russell", "Ruth", "Ryan", "Salvador", "Samantha", "Samuel", "Sandra",
    "Sara", "Sarah", "Scott", "Sean", "Shane", "Sharon", "Shawn", "Sherman", "Sherri",
    "Sherry", "Shirley", "Sonya", "Sophia", "Spencer", "Stacey", "Stella", "Stephanie", "Stephen",
    "Steven", "Susan", "Suzanne", "Sylvia", "Tamara", "Tammy", "Tara", "Teresa", "Terrance",
    "Terry", "Theresa", "Thomas", "Tiffany", "Timothy", "Tina", "Todd", "Tomas", "Tony",
    "Tracy", "Trevor", "Troy", "Tyler", "Vanessa", "Veronica", "Vicki", "Victor", "Victoria",
    "Vincent", "Virginia", "Wade", "Walter", "Warren", "Wayne", "Wendy", "Wesley", "William",
    "Willie", "Wilma", "Yolanda", "Yvette", "Yvonne", "Zachary"
]))


# The student class is used to generate unique students with properties chosen based on desired probability distributions
class Student:
    def __init__(self, number):
        self.number = number
        self.name = np.random.choice(names)
        self.major = np.random.choice(course_departments)
        self.academic_year = np.random.randint(1,5)
        self.age = np.random.randint(18, 50)
        self.courses = self.generate_courses()

    # Each student is given a list of randomly generated courses
    # The list can be 1-6 courses with a probability distribution favoring 5 courses on average
    # Ideally, we would also pick the course_department with probabilities favoring
    #  the same department as the student's major, but it is currently just random
    # Course levels are chosen with probabilities favoring the student's academic_year, then a course_number is chosen randomly
    # The result is a list of courses that could be similar to a real student's courses (aside from the random department choice)
    def generate_courses(self):
      courselist = []
      for course in range(np.random.choice([1,2,3,4,5,6], p=[0.05, 0.1, 0.1, 0.25, 0.4, 0.1])):
        course_levels = [100, 200, 300, 400]
        # array contains 4 different probability distributions, 1 for each academic_year case 1-4
        probabilities = [[0.8, 0.2, 0, 0],
                         [0.2, 0.6, 0.2, 0],
                         [0.1, 0.2, 0.5, 0.2],
                         [0.1, 0.1, 0.3, 0.5]]
        course_1 = np.random.choice(course_departments)
        course_2 = np.random.choice(course_levels, p=probabilities[int(self.academic_year) - 1]) + np.random.choice(course_numbers)
        course =  course_1 + str(course_2)
        courselist.append(course)
      return courselist

    def __str__(self):
        return f"Student_Number: {self.number}\nName: {self.name}\nMajor: {self.major}\nAcademic Year: {self.academic_year}\nAge: {self.age}\nCourses: {', '.join(self.courses)}\n"

# Generate the list of students
students = [Student(number) for number, student in enumerate(range(num_students), 1)]

def print_students():
  for student in students:
   print(student)

# Uncomment the following line to see the list of students generated
print_students()
