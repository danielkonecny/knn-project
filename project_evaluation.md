# Hodnocení projektu

## Návrh bodů
36

## Je pochopitelně definovaná úloha?
Ano

## Vycházejí z něčeho přibližně state-of-the art? Odkazuje se report na něco takového?
Faster R-CNN a ResNet (2015), takže není úplně SOTA.

## Náročnost a vhodnost přístupu?
Použítí hotového detektoru a klasifikátor "čistě" ResNet (obojí dotrénováno). Oddělený detektor a klasifikátor asi není zcela ok. Region proposal detektor rozhodně nebude použitelný realtime (pokud je cílem).

## Vhodnost trénovacích a testovacích dat.
Dataset Mapillary je pro úlohu vhodný.

## Vhodnost způsobu vyhodnocení. Odpovídá zvyklostem v dané oblasti? Pokud používají existující veřejný dataset, odpovídá vyhodnocení standardnímu postupu pro daný dataset?
Způsob asi v pořádku. Chybí porovnání s nějakou již existující metodou. Na datech z ČR vyhodnoceno jen "kouknu a vidím".

## Provedli nějaké systematické experimenty, které vedly k "opPorovnání ResNet50 vs. ResNet101. Jinak spíše "na první dobrou" - žádné větší experimenty vedoucí k lepším výsledkům.timalizaci" přístupu (zlepšení výsledků)?
Porovnání ResNet50 vs. ResNet101. Jinak spíše "na první dobrou" - žádné větší experimenty vedoucí k lepším výsledkům.

## Dá se text číst? Není to typografická hrůza? Odkazují se na zdroje?
Typograficky, až na špatně vložené citace, v pořádku. Citací je ovšem málo (jen 4) a ani nijak úplně vhodné a aktuální (ImageNet, dataset, ResNet, Faster R-CNN).

## Jaký je rozsah a složitost vytvořeného kódu? Pokud používají existující zdrojáky, je pochopitelné, co udělali v rámci projektu a co převzali?
Pravděpodobně implementováno samostatně (bez odkazů). Většinou se jedná o využití stávajících metod - Detectron2.

## Otázky k obhajobě?
Proč jste nezkusili nějaký jednokrokový detektor?
Prováděli jste nějaké kvalntitativní vyhodnocení na značkách z ČR?
