
INTRA_CLAUSE_CONJUNCTIONS = {
    "Enhancement": {
        "Temporal": {
            "Simultaneous": {
                "paratactic": ["and", "meanwhile", "at the same time", "simultaneously"],
                "hypotactic": ["while", "as", "when", "whilst", "just as"]
            },
            "Later": {
                "paratactic": ["then", "afterwards", "subsequently", "later on"],
                "hypotactic": ["after", "since", "once", "when", "as soon as"]
            },
            "Earlier": {
                "paratactic": ["before that", "previously", "earlier", "until then"],
                "hypotactic": ["before", "until", "till", "prior to", "by the time"]
            }
        },
        "Causal-Conditional": {
            "Reason": {
                "paratactic": ["so", "and so", "therefore", "thus", "hence", "consequently"],
                "hypotactic": ["because", "as", "since", "given that", "due to the fact that"]
            },
            "Purpose": {
                "hypotactic": ["so that", "in order that", "that", "lest", "for fear that"]
            },
            "Conditional": {
                "hypotactic": ["if", "unless", "provided that", "assuming that", "in case", "only if"]
            },
            "Concessive": {
                "paratactic": ["but", "yet", "still", "nevertheless", "however", "even so"],
                "hypotactic": ["although", "even though", "while", "though", "despite the fact that"]
            }
        },
        "Manner": {
            "Means": {
                "hypotactic": ["by", "through", "by means of", "via", "using"]
            },
            "Comparison": {
                "paratactic": ["and similarly", "so", "thus", "likewise", "in the same way"],
                "hypotactic": ["as if", "like", "the way", "as though", "in a way that"]
            }
        },
        "Spatial": {
            "Same Place": {
                "paratactic": ["and there", "here", "in that place"],
                "hypotactic": ["where", "wherever", "in which", "at which"]
            },
            "Different Place": {
                "paratactic": ["elsewhere", "in another place"],
                "hypotactic": ["away from", "beyond", "outside of"]
            }
        }
    },
    "Extension": {
        "Addition": {
            "paratactic": ["and", "also", "too", "besides", "furthermore", "moreover"],
            "hypotactic": ["while", "whereas", "along with", "in addition to"]
        },
        "Variation": {
            "Replacive": {
                "paratactic": ["but not", "instead", "rather", "on the contrary"],
                "hypotactic": ["rather than", "instead of", "as opposed to"]
            },
            "Subtractive": {
                "paratactic": ["but", "except", "excluding", "save for"],
                "hypotactic": ["except that", "but that", "other than"]
            }
        },
        "Alternation": {
            "paratactic": ["or", "either ... or", "alternatively", "or else"],
            "hypotactic": ["if not ... then", "whether", "whether ... or not"]
        }
    },
    "Elaboration": {
        "Clarifying": {
            "paratactic": ["that is", "in other words", "namely", "i.e.", "to put it differently"],
            "hypotactic": ["as", "because", "meaning that", "which implies"]
        },
        "Exemplifying": {
            "paratactic": ["for example", "such as", "e.g.", "for instance", "to illustrate"],
            "hypotactic": ["as if", "as though", "in a manner similar to"]
        },
        "Amplification": {
            "paratactic": ["indeed", "in fact", "actually", "specifically"],
            "hypotactic": ["particularly", "notably", "especially"]
        }
    }
}
