"""Pinned character policy for the Mercari source tokenizer adapter.

Character alphabet from mercari/text_utils.py in pjankiewicz/mercari-solution,
commit 374301ad1c32cbc93dcc40313d5d7bb9c5503746.
Copyright (c) 2018 Paweł Jankiewicz, Konstantin Lopuhin; MIT license:
docs/licenses/mercari-solution-LICENSE.txt.

This adapter specifies token boundaries only. It does not implement the
source's separate cleaning, transliteration, stemming or table preprocessing.
"""
from sciona.character_run_tokenizer import CharacterRunTokenizer


WORD_CHARACTERS = '-&0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿĀāĂăĄąĆćĈĉĊċČčĎďĐđĒēĔĕĖėĘęĚěĜĝĞğĠġĢģĤĥĦħĨĩĪīĬĭĮįİıĲĳĴĵĶķĸĹĺĻļĽľĿŀŁłńŅņŇňŉŊŋŌōŎŏŐőŒœŔŕŖŗŘřŚśŜŝŞşŠšŢţŤťŦŧŨũŪūŬŭŮůŰűŲųŴŵŶŷŸŹźŻżŽžſΑΒΓΔΕΖΗΘΙΚΛΜΝΟΠΡΣΤΥΦΧΨΩΪΫάέήίΰαβγδεζηθικλμνξοπρςστυφχψω'


def source_tokenizer():
    return CharacterRunTokenizer(WORD_CHARACTERS, ' \t\n')
