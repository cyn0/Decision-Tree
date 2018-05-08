from scipy.io import arff
import math
import sys

class TreeNode():
    def __init__(self, val):
        self.split_attribute = val
        self.is_leaf = False
        self.is_nominal = True
        self.threshold = 0
        self.attribute_values = []
        self.children = []
        self.classification = ""
        self.class_counts = []

def is_continous_attribute(temp_attribute):
    if str(total_attr_meta[temp_attribute][0]) == 'numeric':
        return True
    else:
        return False


def get_attributes_frequency_map(items, attributes):
    attribute_freq = {}

    for attribute in attributes:
        attribute_freq[attribute] = items.count(attribute)
    return attribute_freq


def get_entropy(rows, class_attributes):
    entropy_value = 0.0
    class_attr = [row[-1] for row in rows]
    class_attribute_freq = get_attributes_frequency_map(class_attr, class_attributes[1])
    row_size = len(rows)

    for key in class_attribute_freq.keys():
        val = class_attribute_freq[key]
        temp = float(val) / row_size
        if temp > 0.0:
            entropy_value += -temp * math.log(temp, 2)
    return entropy_value


def get_numeric_entropy(rows, attribute, threshold, class_attribute):
    index = total_attr.index(attribute)
    lesser = [row for row in rows if row[index] <= threshold]
    greater = [row for row in rows if row[index] > threshold]

    row_size = len(rows)
    left_entropy = (float(len(lesser))/row_size) * get_entropy(lesser, class_attribute)
    right_entropy = (float(len(greater))/row_size) * get_entropy(greater, class_attribute)
    return left_entropy + right_entropy


def get_info_gain(attributes, rows, gain_attribute, class_attribute):
    # print "calculating gain for " + str(gain_attribute)
    # print attributes

    index = total_attr.index(gain_attribute)
    temp_entropy = 0.0
    row_size = len(rows)

    total_entropy = get_entropy(rows, class_attribute)

    if is_continous_attribute(gain_attribute):
        """
        # print index
        sorted_rows = sorted(rows, key=lambda row_item: row_item[index])
        #print json.dumps(rows)
        temp_class_attr = sorted_rows[0][-1]
        prev_val = sorted_rows[0][index]
        i = 0
        uniqueVals = sorted(list(set(zip(*rows)[index])))
        valSets = {key: [] for key in uniqueVals}

        for instance in rows:
            valSets[instance[index]].append((instance[index], instance[-1]))

        candidates = []
        if (len(uniqueVals) > 1):
            for i in range(len(uniqueVals) - 1):
                sLabels = set(zip(*valSets[uniqueVals[i]])[1])
                tLabels = set(zip(*valSets[uniqueVals[i + 1]])[1])
                # if there exists a pair of instances with different labels between the two sets
                if (len(sLabels) == 2 or len(tLabels) == 2 or not (sLabels == tLabels)):
                    candidates.append((uniqueVals[i] + uniqueVals[i + 1]) / float(2))

        # decide which candidate has the best info gain
        bestGain = float("-inf")
        bestCand = None
        if len(candidates) > 0:
            for candidate in candidates:
                candGain = total_entropy - get_numeric_entropy(rows, gain_attribute, candidate, class_attribute)
                if candGain > bestGain:
                    bestGain = candGain
                    bestCand = candidate
        print (bestCand, bestGain)
        return bestGain, bestCand
        """
        # print "$"*100
        # print uniqueVals
        # print "$"*100
        sorted_rows = sorted(rows, key=lambda row_item: row_item[index])
        max_info_gain = 0.0
        threshold = -1

        for i in range(1, len(sorted_rows)):
            row = sorted_rows[i]
            less = []
            my_thr = (sorted_rows[i-1][index] + sorted_rows[i][index]) / 2.0
            left, right = get_filtered_rows_by_threshold(rows, None, gain_attribute, my_thr)
            # print my_thr
            # print left

            # print right
            # left = sorted_rows[:i]
            # right = sorted_rows[i:]

            temp_entropy = 0.0
            temp = float(len(left)) / row_size
            if temp > 0.0:
                temp_entropy = temp * get_entropy(left, class_attribute)

            temp = float(len(right)) / row_size
            if temp > 0.0:
                temp_entropy += temp * get_entropy(right, class_attribute)

            curr_info_gain = total_entropy - temp_entropy

            if curr_info_gain > max_info_gain:
                max_info_gain = curr_info_gain
                threshold = my_thr

        return max_info_gain, threshold

    else:
        attribute_freq = {}

        for row in rows:
            key = row[index]
            if key in attribute_freq:
                attribute_freq[key] += 1.0
            else:
                attribute_freq[key] = 1.0

        for key in attribute_freq.keys():
            temp = float(attribute_freq[key]) / row_size
            rows_excluding_key = [row for row in rows if row[index] == key]
            temp_entropy += temp * get_entropy(rows_excluding_key, class_attribute)

        return total_entropy - temp_entropy, None


def get_split_attribute(attributes, rows, class_attribute_list):
    split_attribute = attributes[0]
    max_gain_so_far = 0.0
    threshold = 0.0

    for attribute in attributes:
        newGain, new_threshold = get_info_gain(attributes, rows, attribute, class_attribute_list)
        # print "Gain for " + attribute + " " +str(newGain)

        print "Gain for %s is %s" % (attribute, str(newGain))
        if newGain > max_gain_so_far:
            max_gain_so_far = newGain
            split_attribute = attribute
            threshold = new_threshold

    return split_attribute, threshold


def get_most_frequent_class_attr(items, class_attributes):
    freq = get_attributes_frequency_map(items, class_attributes[1])

    most_frequent = None
    max = 0

    for key in freq.keys():
        if freq[key] > max:
            max = freq[key]
            most_frequent = key

    return most_frequent

def get_filtered_rows_by_threshold(rows, attributes_list, attribute, threshold):
    index = total_attr.index(attribute)
    less = []
    greater = []
    for row in rows:
        if(row[index] <= threshold):
            less.append(row)
        else:
            greater.append(row)

    return less, greater

def get_filtered_rows_by_attribute(rows, attributes_list, attribute, attribute_value):
    index = total_attr.index(attribute)
    return [row for row in rows if row[index] == attribute_value]


def build_decision_trees(meta_attributes, attributes, rows, class_attribute, m, parent_majority_label = None):
    # print "new node"
    # print "attr"
    # print attributes
    # print rows

    class_attr = [row[-1] for row in rows]

    most_frequent = get_most_frequent_class_attr(class_attr, class_attribute)
    # class_count = get_class_count(class_attr, class_attribute)

    if len(attributes) == 1:
        node = TreeNode("")
        node.is_leaf = True
        node.classification = most_frequent
        node.class_counts = get_class_count_leaf(class_attr, class_attribute)

    elif len(rows) == 0:

        node = TreeNode("")
        node.is_leaf = True
        node.classification = parent_majority_label
        node.class_counts = get_class_count_leaf(class_attr, class_attribute)

    elif len(rows) < m:
        node = TreeNode("")
        node.is_leaf = True
        # node.classification = parent_majority_label
        # node.class_counts = get_class_count_leaf(class_attr, class_attribute)

        count = get_class_count_leaf(class_attr, class_attribute)
        if count[0] == count[1]:
            node.classification = parent_majority_label
        elif count[0] > count[1]:
            node.classification = class_attribute[1][0]
        else:
            node.classification = class_attribute[1][1]

        node.class_counts = count

    elif class_attr.count(class_attr[0]) == len(class_attr):
        node = TreeNode("")
        node.is_leaf = True
        node.classification = most_frequent
        node.class_counts = get_class_count_leaf(class_attr, class_attribute)

    else:
        split_attribute, threshold = get_split_attribute(attributes, rows, class_attribute)
        node = TreeNode(split_attribute)

        # if gain <= 0.0:
        #     if is_continous_attribute(split_attribute):
        #         node.is_nominal = False
        #         node.threshold = threshold
        #     else:
        #         node.is_nominal = True
        #     node.classification = most_frequent
        #     return node

        split_attribute_meta = meta_attributes[split_attribute]
        split_attribute_type, split_attribute_values = split_attribute_meta[0], split_attribute_meta[1]

        if is_continous_attribute(split_attribute):
            node.is_nominal = False
            node.threshold = threshold
            less, greater = get_filtered_rows_by_threshold(rows, attributes, split_attribute, threshold)

            class_count_1 = get_class_count(less, class_attribute)
            class_count_2 = get_class_count(greater, class_attribute)

            child1 = build_decision_trees(meta_attributes, attributes, less, class_attribute, m, most_frequent)
            node.children.append(child1)
            node.class_counts.append(class_count_1)

            child2 = build_decision_trees(meta_attributes, attributes, greater, class_attribute, m, most_frequent)
            node.children.append(child2)
            node.class_counts.append(class_count_2)

        else:
            for val in split_attribute_values:
                filtered_rows = get_filtered_rows_by_attribute(rows, attributes, split_attribute, val)
                modified_attr = attributes[:]
                modified_attr.remove(split_attribute)
                class_count_1 = get_class_count(filtered_rows, class_attribute)
                child = build_decision_trees(meta_attributes, modified_attr, filtered_rows, class_attribute, m, most_frequent)
                node.children.append(child)
                node.attribute_values.append(val)
                node.class_counts.append(class_count_1)

    return node


def get_class_count(rows, class_attributes):
    count_1 = 0
    count_2 = 0
    class_attr = class_attributes[1]

    for row in rows:
        if row[-1] == class_attr[0]:
            count_1 = count_1+1
        else:
            count_2 = count_2+1

    # for attr in class_attributes[1]:
    #     counts.append(rows.count(attr))
    counts = [count_1, count_2]
    return counts


def get_class_count_leaf(rows, class_attributes):
    counts = []
    for attr in class_attributes[1]:
        counts.append(rows.count(attr))

    return counts


def print_tree(depth, node):
    line = ''
    line += depth*'|\t'
    line += node.split_attribute.lower()

    if not node.is_nominal:
        if node.is_leaf:
            pass
        else:
            thr = "%0.6f" % (node.threshold,)
            left_child = node.children[0]
            right_child = node.children[1]

            if left_child.is_leaf:
                l_line = line + " <= " + thr
                l_line += ' [' + ' '.join(str(x) for x in node.class_counts[0]) + ']'
                l_line += ': ' + left_child.classification
                print l_line

            else:
                left_c_count = ' [' + ' '.join(str(x) for x in node.class_counts[0]) + ']'
                print line + " <= " + thr + left_c_count
                print_tree(depth+1, left_child)

            if right_child.is_leaf:
                line += " > " + thr
                line += ' [' + ' '.join(str(x) for x in node.class_counts[1]) + ']' + ': ' + right_child.classification
                print line

            else:
                right_c_count = ' [' + ' '.join(str(x) for x in node.class_counts[1]) + ']'

                print line + " > " + thr + right_c_count
                print_tree(depth+1, right_child)
    else:
        if node.is_leaf:
            pass
        else:
            for i in range(0, len(node.children)):
                child = node.children[i]
                c_count = node.class_counts[i]
                c_count_str = ' [' + ' '.join(str(x) for x in c_count) + ']'
                if child.is_leaf:
                    print line +" = " + node.attribute_values[i] + c_count_str + ": " + child.classification
                else:
                    print line + " = " + node.attribute_values[i] + c_count_str
                    print_tree(depth+1, child)


def classify_row(node, row, attributes):
    if node.is_leaf:
        return node.classification

    split_attribute = node.split_attribute
    index = attributes.index(split_attribute)
    row_value = row[index]

    if node.is_nominal:
        i = node.attribute_values.index(row_value)
        return classify_row(node.children[i], row, attributes)
    else:
        if row_value <= node.threshold:
            return classify_row(node.children[0], row, attributes)
        else:
            return classify_row(node.children[1], row, attributes)


"""
dt-learn <train-set-file> <test-set-file> m
"""
def main():
    args = sys.argv
    # rows, meta = arff.loadarff('credit_train.arff')
    rows, meta = arff.loadarff(args[1])

    attributes_list = meta._attrnames
    attributes = meta._attributes

    class_value = attributes[attributes_list[-1]]
    attributes_list.remove(attributes_list[-1])

    global total_attr, total_attr_meta
    total_attr = attributes_list
    total_attr_meta = attributes

    m = int(args[3])
    root = build_decision_trees(attributes, attributes_list, rows, class_value, m)

    print_tree(0, root)

    # rows, meta = arff.loadarff('credit_test.arff')
    rows, meta = arff.loadarff(args[2])

    correct_prediction = 0
    incorrect_prediction = 0
    print "<Predictions for the Test Set Instances>"
    for i in range(0, len(rows)):
        label = classify_row(root, rows[i], attributes_list)
        actual = rows[i][-1]
        print "%s: Actual: %s Predicted: %s"%(i+1, actual, label)
        if label == actual:
            correct_prediction = correct_prediction + 1
        else:
            incorrect_prediction = incorrect_prediction + 1

    sys.stdout.write("Number of correctly classified: %s Total number of test instances: %s" % (correct_prediction, len(rows)))


if __name__ == "__main__":
    main()
