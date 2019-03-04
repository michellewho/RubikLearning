#!/usr/bin/python3
'''Milestone_A_who_and_what.py
This runnable file will provide a representation of
answers to key questions about your project in CSE 415.

'''

# DO NOT EDIT THE BOILERPLATE PART OF THIS FILE HERE:

CATEGORIES=['Baroque Chess Agent','Feature-Based Reinforcement Learning for the Rubik Cube Puzzle',\
  'Hidden Markov Models: Algorithms and Applications']

class Partner():
  def __init__(self, lastname, firstname, uwnetid):
    self.uwnetid=uwnetid
    self.lastname=lastname
    self.firstname=firstname

  def __lt__(self, other):
    return (self.lastname+","+self.firstname).__lt__(other.lastname+","+other.firstname)

  def __str__(self):
    return self.lastname+", "+self.firstname+" ("+self.uwnetid+")"

class Who_and_what():
  def __init__(self, team, option, title, approach, workload_distribution, references):
    self.team=team
    self.option=option
    self.title=title
    self.approach = approach
    self.workload_distribution = workload_distribution
    self.references = references

  def report(self):
    rpt = 80*"#"+"\n"
    rpt += '''The Who and What for This Submission

Project in CSE 415, University of Washington, Winter, 2019
Milestone A

Team: 
'''
    team_sorted = sorted(self.team)
    # Note that the partner whose name comes first alphabetically
    # must do the turn-in.
    # The other partner(s) should NOT turn anything in.
    rpt += "    "+ str(team_sorted[0])+" (the partner who must turn in all files in Catalyst)\n"
    for p in team_sorted[1:]:
      rpt += "    "+str(p) + " (partner who should NOT turn anything in)\n\n"

    rpt += "Option: "+str(self.option)+"\n\n"
    rpt += "Title: "+self.title + "\n\n"
    rpt += "Approach: "+self.approach + "\n\n"
    rpt += "Workload Distribution: "+self.workload_distribution+"\n\n"
    rpt += "References: \n"
    for i in range(len(self.references)):
      rpt += "  Ref. "+str(i+1)+": "+self.references[i] + "\n"

    rpt += "\n\nThe information here indicates that the following file will need\n"+\
     "to be submitted (in addition to code and possible data files):\n"
    rpt += "    "+\
     {'1':"Baroque_Chess_Agent_Report",'2':"Rubik_Cube_Solver_Report",\
      '3':"Hidden_Markov_Models_Report"}\
        [self.option]+".pdf\n"

    rpt += "\n"+80*"#"+"\n"
    return rpt

# END OF BOILERPLATE.

# Change the following to represent your own information:

michelle = Partner("Ho", "Michelle", "michho")
godwin = Partner("Vincent", "Godwin", "godwinv")
team = [michelle, godwin]

OPTION = '2'
# Legal options are 1, 2, and 3.

title = "Feature-Based Reinforcement Learning for the Rubik Cube "

approach = '''Our approach will be to first understand the rules, then 
develop the problem formulation for the puzzle. After establishing the 
problem, we will implement the learning portion using a feature-based 
approach. In developing the learning portion of solving the Rubik Cube,
we will implement Q learning as well as some heuristics to assist in our search'''

workload_distribution = '''Michelle will primarily be working on the problem formulation, while Godwin begins
implementing some of the heuristics models we want to use. Afterwords, we will verify each other's work and then
proceed to collaborate on the feature based machine learning with Q learning. In addition, both Michelle and Godwin
will work together to optimize efficiency and speed of reaching the goal state of the problem.'''

reference1 = '''Medium article on solving a Rubik's Cube;
    URL: https://medium.com/datadriveninvestor/reinforcement-learning-to-solve-rubiks-cube-and-other-complex-problems-106424cf26ff
     (accessed Mar. 1, 2019)'''

reference2 = '''Chapter 11.3.9.1 SARSA with Linear Function Approximation in Artificial Intelligence: 
    foundations of computational agents, 2nd edition,
    available online at: https://artint.info/html/ArtInt_272.html'''

our_submission = Who_and_what([michelle, godwin], OPTION, title, approach, workload_distribution, [reference1, reference2])

# You can run this file from the command line by typing:
# python3 who_and_what.py

# Running this file by itself should produce a report that seems correct to you.
if __name__ == '__main__':
  print(our_submission.report())