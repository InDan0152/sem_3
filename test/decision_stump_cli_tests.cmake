set(split_data "${WORK_DIR}/decision_stump_cli_split.csv")
file(WRITE "${split_data}"
    "x;target\n0;0\n1;0\n2;0\n3;0\n4;10\n5;10\n6;10\n7;10\n8;10\n9;10\n"
)

execute_process(
    COMMAND "${APP}"
        --input "${split_data}"
        --model decision_stump
        --task regression
        --target target
        --fill zero
        --scale minmax
        --encode ordinal
    RESULT_VARIABLE split_result
    OUTPUT_VARIABLE split_output
    ERROR_VARIABLE split_error
    ENCODING UTF-8
)

if(NOT split_result EQUAL 0)
    message(FATAL_ERROR
        "DecisionStump CLI split scenario failed (${split_result}): ${split_error}"
    )
endif()

foreach(required_text
        "Решающий пень:"
        "Индекс признака:"
        "Порог:"
        "Предсказание левого листа:"
        "Предсказание правого листа:"
        "Результаты регрессии:")
    string(FIND "${split_output}" "${required_text}" text_position)
    if(text_position EQUAL -1)
        message(FATAL_ERROR
            "DecisionStump CLI output is missing '${required_text}': ${split_output}"
        )
    endif()
endforeach()

set(classification_data "${WORK_DIR}/decision_stump_cli_classification.csv")
file(WRITE "${classification_data}"
    "x;target\n0;no\n1;no\n2;no\n3;no\n4;no\n5;yes\n6;yes\n7;yes\n8;yes\n9;yes\n"
)

execute_process(
    COMMAND "${APP}"
        --input "${classification_data}"
        --model decision_stump
        --task classification
        --target target
        --fill zero
        --scale minmax
        --encode ordinal
    RESULT_VARIABLE classification_result
    OUTPUT_VARIABLE classification_output
    ERROR_VARIABLE classification_error
    ENCODING UTF-8
)

if(NOT classification_result EQUAL 0)
    message(FATAL_ERROR
        "DecisionStump CLI classification scenario failed (${classification_result}): ${classification_error}"
    )
endif()

foreach(required_text
        "Решающий пень:"
        "Индекс признака:"
        "Порог:"
        "Предсказание левого листа:"
        "Предсказание правого листа:"
        "Результаты классификации:")
    string(FIND "${classification_output}" "${required_text}" text_position)
    if(text_position EQUAL -1)
        message(FATAL_ERROR
            "DecisionStump CLI classification output is missing '${required_text}': ${classification_output}"
        )
    endif()
endforeach()

set(constant_data "${WORK_DIR}/decision_stump_cli_constant.csv")
file(WRITE "${constant_data}"
    "x;target\n1;0\n1;2\n1;4\n1;6\n1;8\n1;10\n1;12\n1;14\n1;16\n1;18\n"
)

execute_process(
    COMMAND "${APP}"
        --input "${constant_data}"
        --model decision_stump
        --task regression
        --target target
        --fill zero
        --scale minmax
        --encode ordinal
    RESULT_VARIABLE constant_result
    OUTPUT_VARIABLE constant_output
    ERROR_VARIABLE constant_error
    ENCODING UTF-8
)

if(NOT constant_result EQUAL 0)
    message(FATAL_ERROR
        "DecisionStump CLI constant scenario failed (${constant_result}): ${constant_error}"
    )
endif()

foreach(required_text
        "Допустимое разбиение не найдено."
        "Константное предсказание:")
    string(FIND "${constant_output}" "${required_text}" text_position)
    if(text_position EQUAL -1)
        message(FATAL_ERROR
            "DecisionStump CLI output is missing '${required_text}': ${constant_output}"
        )
    endif()
endforeach()

execute_process(
    COMMAND "${APP}"
        --input "${split_data}"
        --model cart
        --task regression
        --target target
        --fill zero
        --scale minmax
        --encode ordinal
        --max-depth 1
    RESULT_VARIABLE cart_result
    OUTPUT_VARIABLE cart_output
    ERROR_VARIABLE cart_error
    ENCODING UTF-8
)

if(NOT cart_result EQUAL 0)
    message(FATAL_ERROR "CART CLI scenario failed (${cart_result}): ${cart_error}")
endif()

string(FIND "${cart_output}" "Структура дерева CART:" cart_summary_position)
string(FIND "${cart_output}" "Решающий пень:" stump_summary_position)
if(cart_summary_position EQUAL -1 OR NOT stump_summary_position EQUAL -1)
    message(FATAL_ERROR "CART CLI output changed unexpectedly: ${cart_output}")
endif()

execute_process(
    COMMAND "${APP}"
        --input "${classification_data}"
        --model cart
        --task classification
        --target target
        --fill zero
        --scale minmax
        --encode ordinal
        --max-depth 1
    RESULT_VARIABLE cart_classifier_result
    OUTPUT_VARIABLE cart_classifier_output
    ERROR_VARIABLE cart_classifier_error
    ENCODING UTF-8
)

if(NOT cart_classifier_result EQUAL 0)
    message(FATAL_ERROR
        "CART classifier CLI scenario failed (${cart_classifier_result}): ${cart_classifier_error}"
    )
endif()

foreach(required_text
        "Структура дерева CART:"
        "Вероятности:"
        "Класс no:")
    string(FIND "${cart_classifier_output}" "${required_text}" text_position)
    if(text_position EQUAL -1)
        message(FATAL_ERROR
            "CART classifier CLI output is missing '${required_text}': ${cart_classifier_output}"
        )
    endif()
endforeach()

string(FIND "${cart_classifier_output}" "Решающий пень:" stump_summary_position)
if(NOT stump_summary_position EQUAL -1)
    message(FATAL_ERROR
        "CART classifier CLI unexpectedly printed stump output: ${cart_classifier_output}"
    )
endif()
