import nltk
from nltk.chunk.regexp import *
from nltk.chunk.util import *

class ChunkerClass():
    def __init__(self):

        chunk_parts = ChunkRule(r"<VB|VBZ|VBP|VBG|VBN|VBD|MD>", "Chunk smaller parts") #chunk smaller parts of chunks
        infinitive = ChunkRule(r"<TO><RB|RBS|RBR>*<VB>", "Chunk infinitive") #chunk infinitive separately

        verb = ChunkRule(r"<DT|JJ|WRB|POS|CD><VB.*><JJ|CD|NN.*>",
                         "Verb assumes adjective function")  #chunk incorrectly tagged, to exclude from chunking

        adverb = ChunkRuleWithContext(r"<MD|TO|VB.*>+", r"<RB|RBS|RBR>+", r"<VB.*>+", "Chunk adverb") #chunk adverb in context with verbs
        cc = ChunkRuleWithContext(r"<VB.*>+", r"<CC>", r"<VB.*>+", "Chunk CC") #chunk and connecting two verbs
        strip_context = StripRule(r"<MD|TO|VB.*>", "Strip context") #strip context from chunkruleswithcontext

        #merge adverb with modal or main verb
        merge_not = MergeRule(r"<MD>", r"<RB|RBS|RBR>+", "Merge won't/wouldn't")  # example: won't
        merge_cc_vb = MergeRule(r"<CC>", r"<VB.*>", "Merge CC and verb")
        merge_rb_vb = MergeRule(r"<RB|RBS|RBR>+", r"<VB.*>", "Merge adverb with a verb")  # example: partly restore

        #merge smaller chunks into more complex chunks
        merge_past_simple = MergeRule(r"<VBD>", r"<RB|RBS|RBR><VB|VBP>", "Merge past simple")  # example: did/VBD n't/RB do/VB
        merge_pres = MergeRule(r"<VB|VBZ|VBP>", r"<RB|RBS|RBR>*<VB|VBZ|VBP|VBG>",
                               "Merge infinitive and present continuous")  # example: 'm/VBZ waiting/VBG

        merge_pres_perf = MergeRule(r"<VB|VBZ|VBP>", r"<RB|RBS|RBR>*<VBN|VBD>", "Merge present perfect")  # example: have/VBZ worked/VBN
        merge_past_perf_cont = MergeRule(r"<VBN|VBD>", r"<RB|RBS|RBR>*<VBN|VBD|VBG>",
                                         "Merge past perfect and past continuous")  # example: had/VBD worked/VBN or was/VBD working/VBG
        merge_pres_past_perf_cont = MergeRule(r"<VB|VBZ|VBP|VBN|VBD><RB|RBS|RBR>?<VBN|VBD><RB|RBS|RBR>?", r"<VBG>",
                                              "Merge present and past perfect continuous")
        # example: have/had been working or hadn't been entirely honest

        merge_going_to = MergeRule(r"<VB|VBZ|VBP><RB|RBS|RBR>*<VBG>", r"<TO><VB>",
                                   "Merge going to-future")
        # example: am/VBZ going/VBG to/TO work/VB
        merge_fut_sim_cont = MergeRule(r"<MD><RB|RBS|RBR>*<VB|VBZ|VBP>", r"<VBD|VBN|VBG>",
                                       "Merge Future Perfect Simple and Continuous")
        # example: will/MD have/VBZ worked/VBN / will/MD be/VB working/VBG
        merge_fut_cont = MergeRule(r"<MD><RB|RBS|RBR>*<VB|VBZ|VBP><VBD|VBN>", r"<VBG>",
                                   "Merge Future Perfect Continuous")
        # example: will/MD have/VBZ been/VBD working/VBG
        merge_future = MergeRule(r"<MD><RB|RBS|RBR>*<VB|VBZ|VBP>?", r"<TO>?<VB|VBZ|VBP>",
                                 "Merge future")
        # example: would/MD help/VB fill/VB / to/TO show/VB
        merge = MergeRule(r"<VBG>", r"<VBD|VBN>", "Merge")
        # example: being given

        merge_with_inf = MergeRule(r"<.*>*<VB.*>", r"<TO><VBP|VB|VBZ>",
                                   "Merge with infinitive")  # example: expected to show

        merge_cc_pres_cont = MergeRule(r"<.*>*<VBG>", r"<CC><VBG>", "Merge two verbs")  # example: waiting and watching
        merge_cc_past_part = MergeRule(r"<.*>*<VBN>", r"<CC><VBN>", "Merge two verbs")  # example: waited and saw
        merge_cc_rest = MergeRule(r"<.*>*<VBP|VB|VBZ>", r"<CC><VB|VBP|VBZ>", "Merge two verbs")  # example: wait and see

        #unchunk priloga, veznika i prideva
        unchunk_unused = UnChunkRule(r"<RB|RBS|RBR|CC>+", "Unchunk unused in VP")
        unchunk_cc_vb = UnChunkRule(r"<CC><VB.*>", "Unchunk CC VB without VB before")
        unchunk_adj = UnChunkRule(r"<DT|JJ|WRB|POS|CD><VB.*><JJ|CD|NN.*>", "Unchunk verb mistaken for adjective")


        #add all rules to one list
        rules = [ adverb, strip_context, cc, verb, infinitive, chunk_parts,
                 merge_rb_vb, merge_not, merge_cc_vb, merge_past_simple,
                 merge_with_inf, merge_pres, merge_pres_perf,
                 merge_past_perf_cont, merge_pres_past_perf_cont, merge_going_to,
                 merge_fut_sim_cont, merge_fut_cont, merge_future, merge,
                 merge_cc_pres_cont, merge_cc_rest, merge_cc_past_part,
                 unchunk_cc_vb, unchunk_unused, unchunk_adj]

        # initialize chunker, use list of rules, label chunks VP
        self.chunker = RegexpChunkParser(rules, chunk_label='VP')
        self.chunk_score = ChunkScore()

    def chunk(self, gold_text, tagged_text):
        for i in range(len(tagged_text)):
            result = self.chunker.parse(tagged_text[i]) #parse each sentence
            self.chunk_score.score(gold_text[i], result) #score each sentence

    def evaluate(self):
        #round to 2 decimals
        precision = round(self.chunk_score.precision()*100, 2) #precision in %
        recall = round(self.chunk_score.recall()*100, 2)
        f_measure = round(self.chunk_score.f_measure()*100, 2)

        #print values
        print("Precision: " + str(precision) + "%")
        print("Recall: " + str(recall) + "%")
        print("F-measure: " + str(f_measure) + "%")




