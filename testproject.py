import ManualStrategy
import experiment1
import experiment2

def author(self):
    """
    Returns the GT username of the student.

    Returns:
        str: The GT username.
    """
    return "pdawoud3"  # Replace with your GT username


def study_group(self):
    """
    Returns a comma-separated string of GT_Name of each member of your study group.

    Returns:
        str: A comma-separated string of GT_Name(s).
    """
    return "pdawoud3"  # Replace with actual study group members' GT usernames

if __name__ == '__main__':
    ManualStrategy.run()
    experiment1.run()
    experiment2.run()