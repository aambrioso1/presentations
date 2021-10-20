# mingus example

# http://bspaans.github.io/python-mingus/doc/wiki/refMingusCoreKeys.html

import mingus.core.notes as notes
import mingus.core.intervals as intervals
import mingus.core.keys as keys

print(f'{notes.note_to_int("B")=}')
print(f'{keys.get_notes("Bb")=}')
print(f'{intervals.third("F#", "B")=}')

"""
def generate_triad(tonic):
	third = intervals.third(tonic, tonic)
	fifth = intervals.fifth(tonic, tonic)
	return [tonic, third, fifth]

print(generate_triad("D"))
"""

