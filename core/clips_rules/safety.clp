; Правила безопасности Project Sofia

(defrule safety-financial-data
    (action (type send-message) (content ?c))
    (test (or
        (str-index "карта" ?c)
        (str-index "CVV" ?c)
        (str-index "пароль" ?c)
    ))
    =>
    (assert (block-action (reason "Финансовые данные запрещены")))
)

(defrule safety-personal-data
    (action (type send-message) (content ?c))
    (test (or
        (str-index "адрес" ?c)
        (str-index "телефон" ?c)
    ))
    =>
    (assert (warn-action (reason "Персональные данные — проверь согласие")))
)
