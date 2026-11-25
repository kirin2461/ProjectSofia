; Правила безопасности Project Sofia (CLIPS 6.4, clipspy 1.0).
;
; Вход — поток токенов сообщения (core/safety_engine.py разбивает текст):
;   (token (pos 0) (kind word)   (text "мой") (dot no))
;   (token (pos 1) (kind word)   (text "cvv") (dot no))
;   (token (pos 2) (kind number) (text "123") (digits 3))
; Слова приходят в нижнем регистре, ё заменена на е; dot = yes, если сразу
; после слова стоит точка (сокращения «ул.», «д.», «кв.»).
;
; Выход — факты verdict; движок берёт block раньше warn, затем по confidence.
;   (verdict (action block|warn) (rule <имя>) (reason "...") (confidence 0.9))

(deftemplate token
    (slot pos (type INTEGER))
    (slot kind (type SYMBOL) (allowed-symbols word number))
    (slot text (type STRING))
    (slot dot (type SYMBOL) (allowed-symbols yes no) (default no))
    (slot digits (type INTEGER) (default 0)))

(deftemplate verdict
    (slot action (type SYMBOL) (allowed-symbols block warn))
    (slot rule (type SYMBOL))
    (slot reason (type STRING))
    (slot confidence (type FLOAT)))

(deffunction has-prefix (?prefix ?word)
    (eq (str-index ?prefix ?word) 1))

; ════════════════════════ BLOCK: платёжные данные ════════════════════════

; «CVV 123», «cvc: 1234», «код безопасности 123»
(defrule financial-cvv
    (or (token (pos ?p) (kind word) (text "cvv" | "cvc" | "cvv2" | "cvc2"))
        (and (token (pos ?k) (kind word) (text "код"))
             (token (pos ?p&:(= ?p (+ ?k 1))) (kind word) (text "безопасности"))))
    (token (pos ?n&:(and (> ?n ?p) (<= ?n (+ ?p 2)))) (kind number) (digits ?d&3|4))
    =>
    (assert (verdict (action block) (rule financial_cvv)
        (reason "Я не могу обрабатывать платёжные данные (CVV/CVC).")
        (confidence 0.95))))

; Номер карты: 13–19 цифр подряд
(defrule financial-card-number
    (token (kind number) (digits ?d&:(and (>= ?d 13) (<= ?d 19))))
    =>
    (assert (verdict (action block) (rule financial_card)
        (reason "Я не могу обрабатывать номера банковских карт.")
        (confidence 0.9))))

; «мой пароль», «моя карта», «мой пин-код» — свои секреты
(defrule financial-secret
    (token (pos ?p) (kind word) (text "мой" | "моя" | "мои" | "моего" | "моей"))
    (token (pos ?q&:(= ?q (+ ?p 1))) (kind word)
           (text "пароль" | "пароля" | "пин-код" | "пинкод" | "карта" | "карты"))
    =>
    (assert (verdict (action block) (rule financial_secret)
        (reason "Я не могу обрабатывать пароли и платёжные данные.")
        (confidence 0.8))))

; ════════════════════════ BLOCK: угрозы насилия ══════════════════════════

(defrule violence-threat
    (token (pos ?p) (kind word)
           (text "убью" | "убей" | "зарежу" | "взорву" | "прикончу" | "изобью"))
    ; «убью время», «убей таракана» — не угроза человеку
    (not (token (pos ?q&:(= ?q (+ ?p 1))) (kind word)
                (text "время" | "времени" | "вечер" | "час" | "таракана" |
                      "комара" | "муху" | "процесс" | "задачу")))
    =>
    (assert (verdict (action block) (rule violence_threat)
        (reason "Я не могу помогать с угрозами насилия.")
        (confidence 0.85))))

(defrule violence-threat-en
    (token (pos ?p) (kind word) (text "kill"))
    (token (pos ?q&:(= ?q (+ ?p 1))) (kind word) (text "you" | "him" | "her" | "them"))
    =>
    (assert (verdict (action block) (rule violence_threat)
        (reason "Я не могу помогать с угрозами насилия.")
        (confidence 0.85))))

; ════════════════ BLOCK: оружие, взрывчатка, наркотики ═══════════════════

(defrule forbidden-howto
    (token (pos ?p) (kind word) (text "как"))
    (token (pos ?v&:(= ?v (+ ?p 1))) (kind word)
           (text "сделать" | "собрать" | "изготовить" | "приготовить" | "синтезировать"))
    (token (pos ?o&:(and (> ?o ?v) (<= ?o (+ ?v 2)))) (kind word)
           (text ?w&:(or (has-prefix "бомб" ?w) (has-prefix "взрывчат" ?w)
                         (has-prefix "взрывн" ?w) (has-prefix "оружи" ?w)
                         (has-prefix "наркотик" ?w) (has-prefix "метамфетамин" ?w)
                         (has-prefix "пистолет" ?w))))
    =>
    (assert (verdict (action block) (rule forbidden_topic)
        (reason "Я не могу обсуждать изготовление оружия, взрывчатки или наркотиков.")
        (confidence 0.8))))

(defrule forbidden-recipe
    (token (pos ?p) (kind word) (text "рецепт"))
    (token (pos ?q&:(= ?q (+ ?p 1))) (kind word)
           (text ?w&:(or (has-prefix "метамфетамин" ?w) (has-prefix "взрывчат" ?w)
                         (has-prefix "наркотик" ?w))))
    =>
    (assert (verdict (action block) (rule forbidden_topic)
        (reason "Я не могу обсуждать изготовление оружия, взрывчатки или наркотиков.")
        (confidence 0.8))))

(defrule forbidden-howto-en
    (token (pos ?p) (kind word) (text "how"))
    (token (pos ?q&:(= ?q (+ ?p 1))) (kind word) (text "to"))
    (token (pos ?r&:(= ?r (+ ?p 2))) (kind word) (text "make" | "build"))
    (token (pos ?o&:(and (> ?o ?r) (<= ?o (+ ?r 2)))) (kind word)
           (text "bomb" | "explosive" | "explosives" | "meth"))
    =>
    (assert (verdict (action block) (rule forbidden_topic)
        (reason "Я не могу обсуждать изготовление оружия, взрывчатки или наркотиков.")
        (confidence 0.8))))

; ════════════════════════ WARN: персональные данные ══════════════════════

; «ул. Ленина», «улица Мира», «проспект Победы», «пер. Садовый»
(defrule personal-address-street
    (or (token (kind word) (text "улица" | "проспект" | "переулок" | "пр-т"
                                    | "бульвар" | "шоссе"))
        (token (kind word) (text "ул" | "пер" | "пр") (dot yes)))
    =>
    (assert (verdict (action warn) (rule personal_address)
        (reason "Похоже на персональные данные (адрес). Проверьте согласие на обработку.")
        (confidence 0.7))))

; «д. 5», «кв. 12», «дом 7»
(defrule personal-address-house
    (token (pos ?p) (kind word) (text "д" | "кв" | "дом" | "квартира"))
    (token (pos ?q&:(= ?q (+ ?p 1))) (kind number))
    =>
    (assert (verdict (action warn) (rule personal_address)
        (reason "Похоже на персональные данные (адрес). Проверьте согласие на обработку.")
        (confidence 0.7))))

; «адрес проживания», «адрес регистрации»
(defrule personal-address-registration
    (token (pos ?p) (kind word) (text "адрес"))
    (token (pos ?q&:(= ?q (+ ?p 1))) (kind word)
           (text "проживания" | "регистрации" | "прописки"))
    =>
    (assert (verdict (action warn) (rule personal_address)
        (reason "Похоже на персональные данные (адрес). Проверьте согласие на обработку.")
        (confidence 0.7))))

; Телефон: 10–12 цифр одним блоком («+7 999 123 45 67»)
(defrule personal-phone
    (token (kind number) (digits ?d&:(and (>= ?d 10) (<= ?d 12))))
    =>
    (assert (verdict (action warn) (rule personal_phone)
        (reason "Похоже на номер телефона — персональные данные.")
        (confidence 0.7))))

; Документ и номер рядом: «паспорт 4510 123456», «СНИЛС: 123-456-789 00»
(defrule personal-id
    (token (pos ?p) (kind word) (text "паспорт" | "паспорта" | "снилс" | "инн"))
    (token (pos ?n&:(and (> ?n ?p) (<= ?n (+ ?p 3)))) (kind number))
    =>
    (assert (verdict (action warn) (rule personal_id)
        (reason "Похоже на документ, удостоверяющий личность (персональные данные).")
        (confidence 0.75))))

; ════════════════════ WARN: медицина и юридические вопросы ═══════════════

(defrule medical-keyword
    (token (kind word)
           (text ?w&:(or (has-prefix "дозировк" ?w) (has-prefix "диагноз" ?w)
                         (eq ?w "пропиши") (eq ?w "пропишите"))))
    =>
    (assert (verdict (action warn) (rule medical_advice)
        (reason "Это медицинская тема. Я не врач — нужна консультация специалиста.")
        (confidence 0.6))))

(defrule medical-dose
    (token (pos ?p) (kind word) (text "доза" | "дозы" | "дозу"))
    (token (pos ?q&:(= ?q (+ ?p 1))) (kind word) (text ?w&:(has-prefix "лекарств" ?w)))
    =>
    (assert (verdict (action warn) (rule medical_advice)
        (reason "Это медицинская тема. Я не врач — нужна консультация специалиста.")
        (confidence 0.6))))

(defrule medical-treat-me
    (token (pos ?p) (kind word) (text "лечить" | "вылечить"))
    (token (pos ?q&:(= ?q (+ ?p 1))) (kind word) (text "меня"))
    =>
    (assert (verdict (action warn) (rule medical_advice)
        (reason "Это медицинская тема. Я не врач — нужна консультация специалиста.")
        (confidence 0.6))))

(defrule medical-pills
    (token (pos ?p) (kind word) (text "какие" | "какое" | "что"))
    (token (pos ?q&:(= ?q (+ ?p 1))) (kind word)
           (text "таблетки" | "лекарство" | "антибиотик"))
    (token (pos ?r&:(= ?r (+ ?q 1))) (kind word) (text "пить" | "принимать"))
    =>
    (assert (verdict (action warn) (rule medical_advice)
        (reason "Это медицинская тема. Я не врач — нужна консультация специалиста.")
        (confidence 0.6))))

(defrule legal-keyword
    (token (kind word) (text ?w&:(has-prefix "юридическ" ?w)))
    =>
    (assert (verdict (action warn) (rule legal_advice)
        (reason "Это юридический вопрос. Я не юрист — нужна консультация специалиста.")
        (confidence 0.6))))

(defrule legal-lawyer
    (token (pos ?p) (kind word) (text "консультация" | "консультацию"))
    (token (pos ?q&:(= ?q (+ ?p 1))) (kind word) (text ?w&:(has-prefix "юрист" ?w)))
    =>
    (assert (verdict (action warn) (rule legal_advice)
        (reason "Это юридический вопрос. Я не юрист — нужна консультация специалиста.")
        (confidence 0.6))))

(defrule legal-court
    (token (pos ?p) (kind word) (text "подать"))
    (token (pos ?q&:(= ?q (+ ?p 1))) (kind word) (text "в"))
    (token (pos ?r&:(= ?r (+ ?q 1))) (kind word) (text "суд"))
    =>
    (assert (verdict (action warn) (rule legal_advice)
        (reason "Это юридический вопрос. Я не юрист — нужна консультация специалиста.")
        (confidence 0.6))))

(defrule legal-bypass
    (token (pos ?p) (kind word) (text "как"))
    (token (pos ?q&:(= ?q (+ ?p 1))) (kind word) (text "обойти"))
    (token (pos ?r&:(= ?r (+ ?q 1))) (kind word) (text "закон"))
    =>
    (assert (verdict (action warn) (rule legal_advice)
        (reason "Это юридический вопрос. Я не юрист — нужна консультация специалиста.")
        (confidence 0.6))))
