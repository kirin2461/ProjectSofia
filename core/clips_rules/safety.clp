; Правила безопасности Project Sofia (CLIPS)
;
; Факт входа: (action (type send-message) (content "..."))
; Факт выхода: (block-action | warn-action | allow-action (reason "..."))

(deftemplate action
    (slot type)
    (slot content))

(deftemplate block-action
    (slot reason))

(deftemplate warn-action
    (slot reason))

(deftemplate allow-action
    (slot reason))

; ─── BLOCK: финансовые данные ──────────────────────────────
(defrule safety-financial-cvv
    (action (type send-message) (content ?c))
    (test (or
        (str-index "CVV" ?c)
        (str-index "cvv" ?c)
        (str-index "CVC" ?c)
        (str-index "код безопасности" ?c)
    ))
    =>
    (assert (block-action (reason "Я не могу обрабатывать платёжные данные (CVV/CVC).")))
)

(defrule safety-financial-card
    (action (type send-message) (content ?c))
    (test (or
        (str-index "карта" ?c)
        (str-index "карту" ?c)
        (str-index "пароль" ?c)
        (str-index "пин-код" ?c)
    ))
    =>
    (assert (block-action (reason "Я не могу обрабатывать платёжные данные и пароли.")))
)

; ─── BLOCK: угрозы насилия ─────────────────────────────────
(defrule safety-violence
    (action (type send-message) (content ?c))
    (test (or
        (str-index "убью" ?c)
        (str-index "зарежу" ?c)
        (str-index "взорву" ?c)
        (str-index "прикончу" ?c)
    ))
    =>
    (assert (block-action (reason "Я не могу помогать с угрозами насилия.")))
)

; ─── WARN: персональные данные ─────────────────────────────
(defrule safety-personal-data
    (action (type send-message) (content ?c))
    (test (or
        (str-index "адрес" ?c)
        (str-index "телефон" ?c)
        (str-index "паспорт" ?c)
        (str-index "СНИЛС" ?c)
    ))
    =>
    (assert (warn-action (reason "Похоже на персональные данные — проверьте согласие.")))
)

; ─── WARN: медицинские/юридические советы ──────────────────
(defrule safety-medical
    (action (type send-message) (content ?c))
    (test (or
        (str-index "дозировк" ?c)
        (str-index "диагноз" ?c)
        (str-index "таблетки" ?c)
    ))
    =>
    (assert (warn-action (reason "Медицинская тема — нужна консультация специалиста.")))
)

(defrule safety-legal
    (action (type send-message) (content ?c))
    (test (or
        (str-index "юридическ" ?c)
        (str-index "юрист" ?c)
        (str-index "в суд" ?c)
    ))
    =>
    (assert (warn-action (reason "Юридический вопрос — нужна консультация специалиста.")))
)
