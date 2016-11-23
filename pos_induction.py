# Graph-based POS-tagging for low resource languages
# Part 2: POS projection & induction

# Alexandra Arkut, Janosch Haber, Muriel Hol, Victor Milewski
# v.0.01 - 2016-11-22

# TODO: Complete with coarse universal tag set
POS_TAGS = ['N', 'V', 'ADJ']
UPDATE_ITERATIONS = 10
NU = 0,000002


class Vertex(object):
    """Vertex of the bilingual graph"""

    def __init__(self, arg):
        super(Vertex, self).__init__()
        self.arg = arg


class PeripheralVertex(Vertex):
    """Peripheral vertex of the target language graph"""

    def __init__(self, middleWord, distribution):
        super(PeripheralVertex, self).__init__()
        self.middleWord = middleWord
        self.distribution = distribution


class Word(object):
    """Word in the target language"""

    def __init__(self, token, tag):
        super(Word, self).__init__()
        self.token = token
        self.tag = tag


# 4 - POS Projection
def POS_projection():
    # Transfer POS tags from the source side of the bilingual graph to the peripheral vertices Vfl of the target language
    projectTags()
    # Propagate labels through the target language graph
    propagateTags()


# Transfer POS tags from the source side of the bilingual graph to the peripheral vertices Vfl of the target language
def projectTags():
    Vfl = collectPeripheralVertices()
    return calculateLabelDistribution(Vfl)


# Propagate labels through the target language graph
def propagateTags():
    Vf = collectInternalVertices()
    for i in range (0,UPDATE_ITERATIONS):
        for vertex in Vf:
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
            count = count + w_i_j * q_m_1[possibleLabel] + NU * U_y #TODO: What are w_i_j and U_y
        gamma[possibleLabel] = count
    return gamma


# Calculates the current kappa value for a vertex
def calculateKappa(vertex):
    kappa = NU
    for neighbor in vertex.neighbors:
        kappa = kappa + w_i_j #TODO: What is w_i_j
    return kappa


# Returns the peripheral vertices of the target language graph
def collectPeripheralVertices():
    return None


# Returns the internal vertices of the target language graph
def collectInternalVertices():
    return None


def calculateLabelDistribution(vfl):
    # generate distribution statistics
    labelAlignments = {}
    for vertex in vfl:
        token = vfl.middleWord
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
        distribution = labelDistribution[vertex.middleWord]
        vertex.distribution = distribution
        # TODO Test: Sum of distribution equals 1?
        probSum = 0
        for entry in distribution:
            probSum = probSum + entry
        if (probSum != 1): print "Distribution of probabilities does not sum to 1!"



if __name__ == "__main__":


