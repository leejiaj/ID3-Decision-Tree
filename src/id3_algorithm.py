# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 16:57:36 2017

@author: leejia
"""
import pandas as pd
import math
import sys
from Tree import Tree

#!/usr/bin/python 
global node_num

#reading data from csv
def read_data_csv(data_file):
    file=open(data_file)
    df=pd.read_csv(file,skip_blank_lines=True).dropna()
    return df;
    
#counting number of instances with the class label  
def classLabels_count(df,x):
    df1 = pd.DataFrame(df)
    classList = ['Class']
    return (df1['Class'] == x).sum()

#calculating entropy    
def calculate_entropy(df):
    df1 = pd.DataFrame(df)
    count_zero = classLabels_count(df1,0)
    count_one = classLabels_count(df1,1)
    total = count_zero + count_one
    if total == 0:
        entropy = 0
        return entropy
    p1 = count_zero / total
    p2 = count_one / total
    if p1 == 0:
        if p2 == 0:
            entropy = 0
        else:
            entropy = -1*p2*math.log(p2,2)
    elif p2 == 0:
        entropy = -1*p1*math.log(p1,2)
    else:
        entropy = (-1*p1*math.log(p1,2)) + (-1*p2*math.log(p2,2))
    return entropy
    
#finding the best attribute
def find_best_attribute(df,attributes_list):
    entropy_parent = calculate_entropy(df)
    count = 0
    for i in range(len(attributes_list)):
        entropy_left_child = calculate_entropy(df[df[attributes_list[i]] == 0])
        entropy_right_child = calculate_entropy(df[df[attributes_list[i]] == 1])
        count_left_examples = (df[attributes_list[i]] == 0).sum()
        count_right_examples = (df[attributes_list[i]] == 1).sum()
        total_examples = count_left_examples + count_right_examples
        weight_left = count_left_examples / total_examples
        weight_right = count_right_examples / total_examples
        average_entropy_children = (weight_left*entropy_left_child)+(weight_right*entropy_right_child)
        information_gain = entropy_parent - average_entropy_children
        count = count + 1
        if count > 1:
            if information_gain > max_information_gain:
                selected_attribute = attributes_list[i]
                max_information_gain = information_gain
        else:
            selected_attribute = attributes_list[i]
            max_information_gain = information_gain
    return selected_attribute
    
#finding the example subset for the attribute value    
def find_example_subset(df,selected_attribute,x):
    df1 = df[df[selected_attribute] == x]
    return df1

#finding the size of example subset
def find_size_example_subset(df):
    size_subset = df.shape[0]
    return size_subset

#id3 algorithm implementation
def id3(df,tree,attributes_list,selected_attribute):
    global root
    global node_num
    
    zero_example_count = classLabels_count(df,0)
    one_example_count = classLabels_count(df,1)
            
    #If all examples are positive, Return the single-node tree Root, with label = +.
    if (one_example_count == df.shape[0]):
        tree.data = 1
        tree.left = None
        tree.right = None
        tree.negativecount = 0
        tree.positivecount = one_example_count
        if(root.data == None):
            root = tree      
        return
    
    #If all examples are negative, Return the single-node tree Root, with label = -.
    if(zero_example_count == df.shape[0]):
        tree.data = 0
        tree.left = None
        tree.right = None
        tree.negativecount = zero_example_count
        tree.positivecount = 0
        if(root.data == None):
            root = tree  
        return
    
    #If number of predicting attributes is empty, then Return the single node tree Root,
    #with label = most common value of the target attribute in the examples.
    if len(attributes_list) == 0:
        if zero_example_count >= one_example_count:
            tree.data = 0
        else:
            tree.data = 1
        tree.left = None
        tree.right = None
        tree.negativecount = zero_example_count
        tree.positivecount = one_example_count
        if(root.data == None):
            root = tree  
        return


    # Finding the Attribute that best classifies examples.
    selected_attribute = find_best_attribute(df,attributes_list)
    tree.data = selected_attribute
    
    tree.negativecount = zero_example_count
    tree.positivecount = one_example_count
    if (root.data == None):
        root = tree
    attributes_list1 = attributes_list[:]
    attributes_list2 = attributes_list[:]

    # finding subset of examples that have value 0 for selected attribute
    df1 = find_example_subset(df,selected_attribute,0)
    size = find_size_example_subset(df1)
    #Checking if Examples(0) is empty
    if (size == 0):
        #If empty, adding leaf node with label = most common target value in the examples
        treeLeft = Tree()
        if zero_example_count >= one_example_count:
            treeLeft.data = 0
        else:
            treeLeft.data = 1
        treeLeft.left = None
        treeLeft.right = None
        treeLeft.negativecount = 0
        treeLeft.positivecount = 0
        tree.left = treeLeft
        return
    else:
        #if not empty, add subtree ID3
        treeLeft = Tree()
        tree.left = treeLeft
        attributes_list1.remove(selected_attribute)
        id3(df1,treeLeft,attributes_list1,selected_attribute)
    

    # finding subset of examples that have value 1 for selected attribute
    df2 = find_example_subset(df,selected_attribute,1)
    size = find_size_example_subset(df2)
    #Checking if Examples(1) is empty
    if (size == 0):
        #If empty, adding leaf node with label = most common target value in the examples
        treeRight = Tree()
        if zero_example_count >= one_example_count:
            treeRight.data = 0
        else:
            treeRight.data = 1
        treeRight.left = None
        treeRight.right = None
        treeRight.negativecount = 0
        treeRight.positivecount = 0
        tree.right = treeRight
        return
    else:
        #if not empty, add subtree ID3
        treeRight = Tree()
        tree.right = treeRight  
        attributes_list2.remove(selected_attribute)
        id3(df2,treeRight,attributes_list2,selected_attribute)
    return
    
#printing the tree
def printTree(node,count):
    if node:
        if node.data == 0 or node.data == 1:
            print(" : ",node.data,end="")
        else:   
            print("")
            i = 0
            while i < count:
                print("| ",end="")
                i = i + 1
            print(node.data,end="")
            print(" = 0 ",end="")
        count = count + 1
        printTree(node.left,count)
        if node.data != 0 and node.data != 1:
            print("")
            i = 0
            while i < count-1:
                print("| ",end="")
                i = i + 1
            print(node.data,end="")
            print(" = 1 ",end="")
        printTree(node.right,count)

#numbering the nodes of the tree        
def addNodeNumbersTree(node):
    if node:
        global node_num
        node.nodenum = node_num
        node_num = node_num + 1
        addNodeNumbersTree(node.left)
        addNodeNumbersTree(node.right)
 
#testing the tree with instances       
def testTree(instance,node):
    if node:
        if node.data == 0 or node.data == 1:
            if node.data == 0:
                return 0
            else:
                return 1
        elif instance[node.data] == 0:
            return testTree(instance,node.left)
        elif instance[node.data] == 1:
            return testTree(instance,node.right)
    
#computing accuracy of the model using datasets
def computeAccuracy(data_file):
    global root
    correctlyClassifiedInstanceCount = 0
    total_count = 0
    file=open(data_file)
    df=pd.read_csv(file,skip_blank_lines=True).dropna()
    for index, row in df.iterrows():
        predicted_value = testTree(row,root)
        if row['Class'] == predicted_value:
            correctlyClassifiedInstanceCount = correctlyClassifiedInstanceCount + 1
        total_count = total_count + 1
    accuracy = (correctlyClassifiedInstanceCount / total_count)*100
    return accuracy

#counting the number of nodes on the tree
def countNodesTree(node):
    if node:
        global countNodes
        countNodes = countNodes + 1
        countNodesTree(node.left)
        countNodesTree(node.right)

#counting the number of leaf nodes on the tree
def countLeafNodesTree(node):
    if node:        
        global countLeafNodes
        if node.data == 0 or node.data == 1:
            countLeafNodes = countLeafNodes + 1
        countLeafNodesTree(node.left)
        countLeafNodesTree(node.right)

#printing the Summary before pruning
def printSummaryPrePrune():
    global training_df
    global validation_df
    global test_df
    global node_num
    global root
    global countNodes
    countNodes = 0
    global countLeafNodes
    countLeafNodes = 0
    global training_dataset
    global validation_dataset
    global test_dataset
    
    countTrainingInstances = training_df.shape[0]
    countValidationInstances = validation_df.shape[0]
    countTestInstances = test_df.shape[0]
    
    countTrainingAttributes = training_df.shape[1] - 1
    countValidationAttributes = validation_df.shape[1] - 1
    countTestAttributes = test_df.shape[1] - 1
    
    countNodesTree(root)
    countLeafNodesTree(root)
    
    training_accuracy = computeAccuracy(training_dataset)
    validation_accuracy = computeAccuracy(validation_dataset)
    test_accuracy = computeAccuracy(test_dataset)
    
    print("")
    print("")
    print("Pre-Pruned Accuracy")
    print("-------------------------------------")
    print("Number of training instances = ",countTrainingInstances)
    print("Number of training attributes = ",countTrainingAttributes)
    print("Total number of nodes in the tree = ",countNodes)
    print("Number of leaf nodes in the tree = ",countLeafNodes)
    print("Accuracy of the model on the training dataset = ",training_accuracy,"%")
    print("")
    print("Number of validation instances = ",countValidationInstances)
    print("Number of validation attributes = ",countValidationAttributes)
    print("Accuracy of the model on the validation dataset before pruning = ",validation_accuracy,"%")
    print("")
    print("Number of testing instances = ",countTestInstances)
    print("Number of testing attributes = ",countTestAttributes)
    print("Accuracy of the model on the testing dataset = ",test_accuracy,"%")
    
 
#implementation of pruning
def pruneTreeImplement(node):
    global countPruneNodes
    global nodeNumber
    if node:           
        if (node.nodenum == nodeNumber):
            if node.left and node.right: 
                if (node.left.data == 0 or node.left.data == 1 or node.right.data == 0 or node.right.data == 1 ):
                    if node.negativecount > node.positivecount:
                        node.data = 0
                    else:
                        node.data = 1
                    node.left = None
                    node.right = None
                    countPruneNodes = countPruneNodes - 1
                    return
                else:
                    return
            else:
                return
        else:
            pruneTreeImplement(node.left)
            pruneTreeImplement(node.right)

#pruning the tree           
def pruneTree():
    global root
    global pruning_factor
    global countPruneNodes
    global nodeNumber 
    global countNodes
    countNodes = 0
    countNodesTree(root)
    nodeNumber = countNodes
    countPruneNodes = int(float(pruning_factor) * countNodes)

    while ( countPruneNodes > 0 and nodeNumber > 0):
        pruneTreeImplement(root)
        nodeNumber = nodeNumber - 1

#printing the summary after Pruning
def printSummaryPostPrune():
    global training_df
    global validation_df
    global test_df
    global node_num
    global root
    global countNodes
    countNodes = 0
    global countLeafNodes
    countLeafNodes = 0
    global training_dataset
    global validation_dataset
    global test_dataset
    
    countTrainingInstances = training_df.shape[0]
    countValidationInstances = validation_df.shape[0]
    countTestInstances = test_df.shape[0]
    
    countTrainingAttributes = training_df.shape[1] - 1
    countValidationAttributes = validation_df.shape[1] - 1
    countTestAttributes = test_df.shape[1] - 1
    
    countNodesTree(root)
    countLeafNodesTree(root)
    
    training_accuracy = computeAccuracy(training_dataset)
    validation_accuracy = computeAccuracy(validation_dataset)
    test_accuracy = computeAccuracy(test_dataset)
    
    print("")
    print("")
    print("Post-Pruned Accuracy")
    print("-------------------------------------")
    print("Number of training instances = ",countTrainingInstances)
    print("Number of training attributes = ",countTrainingAttributes)
    print("Total number of nodes in the tree = ",countNodes)
    print("Number of leaf nodes in the tree = ",countLeafNodes)
    print("Accuracy of the model on the training dataset = ",training_accuracy,"%")
    print("")
    print("Number of validation instances = ",countValidationInstances)
    print("Number of validation attributes = ",countValidationAttributes)
    print("Accuracy of the model on the validation dataset after pruning = ",validation_accuracy,"%")
    print("")
    print("Number of testing instances = ",countTestInstances)
    print("Number of testing attributes = ",countTestAttributes)
    print("Accuracy of the model on the testing dataset = ",test_accuracy,"%")   
        
        
if __name__ == '__main__':
    global training_dataset 
    training_dataset = sys.argv[1:][0]
    global validation_dataset 
    validation_dataset = sys.argv[1:][1]
    global test_dataset 
    test_dataset = sys.argv[1:][2]
    global pruning_factor
    pruning_factor = sys.argv[1:][3]
    global root
    global node_num
    root = Tree()
    global training_df
    global validation_df
    global test_df
    training_df = read_data_csv(training_dataset)
    validation_df = read_data_csv(validation_dataset)
    test_df = read_data_csv(test_dataset)
    attributes_list = list(training_df)
    attributes_list.remove('Class')
    node_num = 1
    selected_attribute = 0
    
    #creating decision tree
    id3(training_df,root,attributes_list,selected_attribute) 
    
    #numbering nodes of the tree
    addNodeNumbersTree(root)
    
    #printing the tree before pruning
    print("")
    print("Decision Tree before Pruning")
    print("---------------------------")
    countforFormatting = 0
    printTree(root,countforFormatting)
    
    #printing the summary before Pruning
    printSummaryPrePrune()
    
    #pruning the tree
    pruneTree()

    print("-----------------------------")
    print("Pruning Factor : ",pruning_factor)
    print("-----------------------------")
    
    #printing the tree after pruning    
    print("Decision Tree after Pruning")
    print("---------------------------")
    countforFormatting = 0
    printTree(root,countforFormatting)
    
    #printing the summary after Pruning
    printSummaryPostPrune()
    
    


