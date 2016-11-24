# Graph-based POS-tagging for low resource languages
# Part 4+5: POS projection & POS induction

# Alexandra Arkut, Janosch Haber, Muriel Hol, Victor Milewski
# v.0.4 - 2016-11-22

POS_TAGS = ['NOUN', 'VERB', 'ADJ', 'ADV', 'PRON', 'DET', 'ADP', 'NUM', 'CONJ', 'PRT', 'PUNC', 'X']
NUMBER_TAGS = len(POS_TAGS)

UPDATE_ITERATIONS = 10
NU = 0,000002
TAU = 0.02

UNIFORM_PROB = 1/NUMBER_TAGS
UNIFORM = {}
for tag in POS_TAGS:
    UNIFORM[tag] = UNIFORM_PROB


class SourceVertex(object):
    """Vertex of the source language graph"""

    def __init__(self, token, tag):
        super(SourceVertex, self).__init__()
        self.token = token
        self.tag = tag


class TargetVertex(object):
    """Vertex of the target language graph"""

    def __init__(self, trigram):
        super(TargetVertex, self).__init__()
        self.token = trigram[1]
        self.left = trigram[0]
        self.right = trigram[2]

        # list of pointers to the 5 vertices with the highest feature similarity score.
        self.neighbors = []
        # dictionary that takes a neighbor pointer as key and the similarity feature measure
        self.weights = {}
        # dictionary that takes the 12 universal tags as keys and the corresponding probability as value
        self.tag_distribution = {}

class PeripheralTargetVertex(TargetVertex):
    """Peripheral vertex of the target language graph"""

    def __init__(self, targetVertex, sourceVertex):
        super(PeripheralTargetVertex, self).__init__()
        self.token = targetVertex.token
        self.left = targetVertex.left
        self.right = targetVertex.right

        # list of pointers to the 5 vertices with the highest feature similarity score.
        self.neighbors = []
        # dictionary that takes a neighbor pointer as key and the similarity feature measure
        self.weights = {}
        # dictionary that takes the 12 universal tags as keys and the corresponding probability as value
        self.tag_distribution = {}
        # pointer to the source language word that the middle word is algined to
        self.aligned = sourceVertex


# 4 - POS Projection
def POS_projection(graph):
    # Transfer POS tags from the source side of the bilingual graph to the peripheral vertices Vfl of the target language
    projectTags(graph)
    # Propagate labels through the target language graph
    propagateTags(graph)


# 5 - POS Induction
def POS_induction():
    # Compute tag probabilities for foreign word types
    wordTypeTagProbs = calcTagProbs()
    # Remove tags with below-threshold probablities
    possibleTags = threshold(wordTypeTagProbs)
    # Determine word features
    wordFeatures = determineFeatures(possibleTags)
    # Induce tags for target language vertices
    runInductionModel()


# Transfers POS tags from the source side of the bilingual graph to the peripheral vertices Vfl of the target language
def projectTags(graph):
    vfl = collectPeripheralVertices(graph)
    return calculateLabelDistribution(vfl)


# Propagates labels through the target language graph
def propagateTags(graph):
    vf = collectInternalVertices(graph)
    for i in range (0,UPDATE_ITERATIONS):
        for vertex in vf:
            gamma = calculateGamma(vertex)
            kappa = calculateKappa(vertex)
            vertex.distribution = gamma / kappa


# Calculates the current gamma distribution for a vertex
def calculateGamma(vertex):
    q_m_1 = vertex.distribution
    gamma = {}
    for possibleLabel in POS_TAGS:
        count = 0
        for neighbor in vertex.neighbors:
            count = count + vertex.weight[neighbor] * q_m_1[possibleLabel] + NU * UNIFORM_PROB
        gamma[possibleLabel] = count
    return gamma


# Calculates the current kappa value for a vertex
def calculateKappa(vertex):
    kappa = NU
    for neighbor in vertex.neighbors:
        kappa = kappa + vertex.weights[neighbor]
    return kappa


# Returns the peripheral vertices of the target language graph
def collectPeripheralVertices(graph):
    periheralVertices = []
    for vertex in graph:
        if vertex.__class__.__name__ == PeripheralTargetVertex: #TODO: Check if this actually works!
            periheralVertices.append(vertex)
    return periheralVertices


# Returns the internal vertices of the target language graph
def collectInternalVertices(graph):
    internalVertices = []
    for vertex in graph:
        if vertex.__class__.__name__ != PeripheralTargetVertex: #TODO: Check if this actually works!
            internalVertices.append(vertex)
    return internalVertices


def calculateLabelDistribution(vfl):
    # generate distribution statistics
    labelAlignments = {}
    for vertex in vfl:
        token = vfl.token
        alignedWord = vertex.alignedWord
        sourceTag = alignedWord.tag
        if token in labelAlignments:
            tagCounts = labelAlignments[token]
            if sourceTag in tagCounts:
                count = tagCounts[sourceTag]
                tagCounts.update(sourceTag, count + 1)
            else:
                tagCounts[sourceTag] = 1
        else:
            labelAlignments[token] = dict(sourceTag = 1)

    # calculate label distribution for all words occurring in peripheral vertices
    labelDistribution = {}
    for token in labelAlignments:
        labelDistribution[token] = {}
        for possibleLabel in POS_TAGS:
            if possibleLabel in labelAlignments[token]:
                countAlignment = labelAlignments[token][possibleLabel]
                countOtherAlignments = -countAlignment
                for tag in POS_TAGS:
                    countOtherAlignments = countOtherAlignments + labelAlignments[token][tag]

                labelDistribution[token][possibleLabel] = countAlignment / countOtherAlignments
            else:
                labelDistribution[token][possibleLabel] = 0

    # assign label distributions to peripheral vertices
    for vertex in vfl:
        distribution = labelDistribution[vertex.token]
        vertex.distribution = distribution

        # TODO Test: Sum of distribution equals 1?
        probSum = 0
        for entry in distribution:
            probSum = probSum + entry
        if (probSum != 1): print "Distribution of probabilities does not sum to 1!"


# Calculates the distribution of tags for all word types encountered in the target language
def calcTagProbs(vf):
    dictionary = {}
    for vertex in vf:
        wordType = vertex.token
        distribution = vertex.distribution
        if wordType in dictionary:
            tokenDistribution = dictionary[wordType]
            dictionary[wordType] = {k: tokenDistribution.get(k, 0) + distribution.get(k, 0) \
                                    for k in set(tokenDistribution) & set(distribution)}
        else:
            dictionary[wordType] = distribution

    wordTypeTagProbs = {}
    for wordType in dictionary:
        wordTypeTagProbs[wordType] = {}
        for tag in dictionary[wordType]:
            tagProb = dictionary[wordType][tag]
            distSum = sum(dictionary[wordType]) - tagProb
            wordTypeTagProbs[wordType][tag] = tagProb / distSum

    return wordTypeTagProbs


# Returns a dictionary with possible tags for all word types by thresholding the tag distributions
def threshold(wordTypeTagProbs):
    possibleTags = {}
    for wordType in wordTypeTagProbs:
        possibleTags[wordType] = {}
        for tag in wordTypeTagProbs[wordType]:
            if wordTypeTagProbs[wordType][tag] > TAU:
                possibleTags[wordType][tag] = 1
            else:
                possibleTags[wordType][tag] = 0
    return possibleTags


# Runs the POS induction model
def runInductionModel():
    # TODO: Implement induction model
    return None


# Extends the vector of possible tags with features of the word type
def determineFeatures(possibleTags):
    for wordType in possibleTags:
        features = dict2vec(possibleTags[wordType])
        if wordIdentity(wordType):
            features.append(1)
        else: features.append(0)
        if containsDigit(wordType):
            features.append(1)
        else: features.append(0)
        if isCap(wordType):
            features.append(1)
        else: features.append(0)
        if containsHyphen(wordType):
            features.append(1)
        else: features.append(0)
        features.extend(checkSuffix(wordType))
        possibleTags[wordType] = features
    return possibleTags


# Transforms a given dictionary into a vector
def dict2vec(dictionary):
    vector = []
    for key in dictionary:
        vector.append(dictionary[key])
    return vector


# Transforms a given vector into a dictionary with the keys specified
def vec2dic(vector, keys):
    if (len(vector) == len(keys)):
        dictionary = {}
        for i in range(len(keys)):
            dictionary[keys[i]] = vector[i]
        return dictionary
    else:
        print "ERROR: Vector size does not match number of keys"
        return None


# Returns True in case of word identity, False otherwise
def wordIdentity(word):
    # TODO! Check word identity?!
    return False


# Returns True if word contains a digit, False otherwise
def containsDigit(word):
    if '.' in word: return True
    else: return False


# Returns True if word starts with an uppercase letter, False otherwise
def isCap(word):
    if word[0].isupper(): return True
    else: return False


# Returns True if word contains a hyphen, False otherwise
def containsHyphen(word):
    if '-' in word: return True
    else: return False


# Returns a 3x1 vector that has ones for either suffix-1, suffix-2 or suffix-3
def checkSuffix(word):
    #TODO! Check suffix features up to length 3
    return [1, 1, 1]