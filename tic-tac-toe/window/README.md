# Specifikation

## Inledning
Jag tänkte programmera en grafisk applikation för "Conway's Game of Life". Det är ett sätt att från väldigt enkla regler i en grid world kunna simulera något som liknar liv. GUIn ska innehålla en interaktiv miljö där man själv kan ändra spellogiken och skapa/döda celler.

Den största utmaningen kommer att bli att få alla delar av programmet (meny, spelvärld, och drawing canvas) att kommunicera med varandra utan fel


## Användarscenarier

### Test
Axel startar programmet och blir välkomnad av ett användergränssnitt med en stor ritad gridworld. Han klickar på en step-knapp och ser hur cellerna i gridvärlden ändras en iteration i Conways Game of Life och ritas upp på ritytan.


### Ändra regler
Axel använder de fyra textfälten för att uppdatera spellogiken i världen. Där matar han in värden för överbefolkning, svält, gränsen för cellfödelsen, och hur många steg världen ska ta i en iteration. Därefter klickar han på en knapp för att uppdatera spelvärlden med de nya reglerna.

### Skapa egna världar
Axel skapar sin egen värld genom att han interaktivt använder sin muspekare för att direkt skapa eller döda celler i spelrutan på användargränssnittet. När han har skapat sin värld väljer han att simulera den igen med hjälp av knapparna som för världen vidare en generation.

# Kodskelett
```
class GameWorld:
    '''The GameWorld encorporates every concept from game logic, to game rule 
       creation, to coordinate loading'''
    def set_from_coords(self, coords):
        '''Takes in a list of cell coordinates and adds them to the game world'''
        pass

    def step(self):
        '''Steps through and changes the game world specified by the current game rules.'''
        pass


    def reset(self):
        '''Resets game world to no livivng cells'''
        pass

    
    def update_rules(self, starve, overpop, birth, stride, new_size):
        '''Takes in the starve, overpopulation, birthing, stride, and game board size rules and updates the objects game logic values'''
        pass


    def __str__(self):
        '''Converts the game world into a visually appealing string'''
        pass


class DrawArea(Canvas):
    '''A drawing canvas which has acces to the game world 
       so that it can draw and change it through the user gui'''
    def mb1_callback(self, event):
        '''Callback that converts mouse click event coordinates to a change in game world'''
        pass
        

    def draw(self):
        '''Draws game world on canvas'''
        pass


class Window(Frame):
    '''User GUI which lets the user interact with a graphical display of the game world and change the game logic at any time'''
    def initUI(self):
        '''Initializes the graphical interface including the buttons, labels, canvas, and entries'''
        pass

    def _step_callback(self):
        '''Callback function for button that steps through the game world'''
        pass
    

    def _coord_callback(self):
        '''Callback function for button that reads coordinates from file and loads them'''
        pass

    
    def _rules_callback(self):
        '''Callback function for button that reads game rules through GUI and loads them'''
        pass

    
    def _animate_callback(self):
        '''Callback function for button that animates the game world'''
        pass
```
# Programflöde och dataflöde
Programmet börjar med att skapa ett Window-objekt som är grunden för det grafiska gränssnittet. Det initialiseras av att skapa ett GameWorld-objekt (spelvärlden) och DrawArea-objektet (ytan som ritas på), tillsammans med tre listor av button-, entry-, och label-widgets. GameWorld-objektet inehåller en spelvärld som är en tvådimensionell numpy-array av booleans där array[y, x] är cell i position (y, x). Klassen innehåller step-metoden som ändrar värld-arrayen en iteration, en metod som tar in nya värden för spellogik och ändrar klassvärdena, och en metod som tar in en lista av cellkoordinater och ändrar spelvärlden inplace. DrawArea tar in en spelvärld och använder den i sin draw-metod och ritar ut cellen i en färg som beror på cellvärdet i spelvärlden. Den har dessutom en mouse button callback som gör om pixelkoordinater till spelvärldskoordinater och ändrar spelvärlds-arrayen. Efter Window-objektet skapats beror programflödet på använderns input. All programflöde körs genom callback-fuktioner för knapparna i GUIt och callback-funktionen för ritytan. _step_callback anropar GameWorlds step-metod och därefter ritytans draw-metod. _coord_callback anropar en funktion som tar in ett filnamn och läser koordinater från en txt-fil. Därefter anropas GameWorlds set_from_coords-metod med koordinaterna som inparameter. _rules_callback läser värdena från Windows lista av entry-widgets och matar in värdena som inparameter till GameWorlds update_rules-metod. _animate_callback anropar ritytans draw-metod och gameworlds step-metod i en forloop tillsammans med anrop till time.sleep.