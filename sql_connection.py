from numpy import empty
import pymssql 
import numpy as np

from PARAMETERS import SQL_SERVER
from PARAMETERS import SQL_DATABASE_NAME
from PARAMETERS import SQL_PORT

class SQLConnection:
    '''
    Module used to establish saving and reading SQL database for application environment
    '''
    
    def __init__(self,debug=False):
        # Initialize SQL connection
        self.current_scan_id = -1
        self.debug = debug
        self.conn = pymssql.connect(server = SQL_SERVER, 
        database=SQL_DATABASE_NAME, 
        port=SQL_PORT)
        self.cursor = self.conn.cursor()

    def create_scan(self,scan_name,path):
        '''
        Check if the scan with the passed name alredy exist. 
        If yes return -1 to signalize error.
        If not retun scan id.
        '''
        self.cursor.execute('SELECT * FROM SKAN WHERE nazwa=%s', scan_name)
        skan = self.cursor.fetchall()
        if skan:
            print(f"Scan with \"{scan_name}\" name exist. Please input other name.")
            self.current_scan_id = -1
            exit()
        else:
            print(f"Scan with \"{scan_name}\" name doesn't exist.")
            insertStatement = f"INSERT INTO SKAN (nazwa, path) output Inserted.id VALUES (\'{scan_name}\',\'{path}\')" 
            self.cursor.execute(insertStatement)
            tempid = self.cursor.fetchall()
            self.cursor.execute('SELECT * FROM SKAN WHERE nazwa=%s', scan_name)
            skan = self.cursor.fetchall()
            print(f"Created scan with name \"{scan_name}\".")
            self.current_scan_id = tempid[0][0]
    
    def add_tooth(self,tooth_name,d):
        '''
        Take image name as input argument in format {section_number}_{row_nr}.png ex: 1_34.png.
        Add to existing row or create new one.
        Fill tooth table with data.
        '''
        if(self.debug): print(f"Add tooth {tooth_name}")
        row_number = (tooth_name.split('.')[0]).split('_')[1]
        section_number = (tooth_name.split('.')[0]).split('_')[0]
        row_id = self.get_row_id(row_number)
        
        self.cursor.execute('SELECT tooth_number FROM TOOTH WHERE row_id=%d and tooth_number=%d',(row_id,section_number))
        tooth_number = self.cursor.fetchall()
        if tooth_number:
            print("\tError. Tried to initialize the same tooth again.")
            return -1
        else:
            insertStatement = """INSERT INTO TOOTH (row_id, tooth_number, length, width, centre_lenght, centre_width, 
                            stepienie, num_instances, narost, zatarcie, wykruszenie, image_name, score, pred_class) VALUES 
                            ({},{},{},{},{},{},{},{},{},{},{},\'{}\',\'{}\',\'{}\')""".format(row_id, section_number, d['length'], 
                            d['width'], d['centre_lenght'], d['centre_width'], d['stepienie'], 
                            d['num_instances'], d['narost'], d['zatarcie'], d['wykruszenie'], d['image_name'], d['score'], d['pred_class'])
            
            if d['pred_class'] is None:
                insertStatement = """INSERT INTO TOOTH (row_id, tooth_number, length, width, centre_lenght, centre_width, 
                                stepienie, num_instances, narost, zatarcie, wykruszenie, image_name, score, pred_class) VALUES 
                                ({},{},{},{},{},{},{},{},{},{},{},\'{}\',{},{})""".format(row_id, section_number, d['length'], 
                                d['width'], d['centre_lenght'], d['centre_width'], d['stepienie'], 
                                d['num_instances'], d['narost'], d['zatarcie'], d['wykruszenie'], d['image_name'], d['score'], d['pred_class'])                

            insertStatement = insertStatement.replace('None','Null')
            try:
                self.cursor.execute(insertStatement)
                if(self.debug): print(f'\tCreated tooth {tooth_name}.')
                self.conn.commit()
            except:
                print("\t Execution error:")
                print('\t',insertStatement)
                self.cursor.execute(insertStatement)
                if(self.debug): print(f'\tCreated tooth {tooth_name}.')
                self.conn.commit()


            return 0

    def get_scan_param(self, param_name, scan_name = ''):
        if(scan_name):
            insertStatement = f'SELECT {param_name} FROM SKAN WHERE nazwa=\'{scan_name}\';'
            if(self.debug): print(insertStatement)
            self.cursor.execute(insertStatement)
        else:
            self.cursor.execute(f'SELECT {param_name} FROM SKAN;')
        data = self.cursor.fetchall()
        return data

    def get_row_id(self,row_number):
        '''
        Check if the following row exists. 
        If yes - return its id. 
        If no - create it and add return its id after that.
        '''
        self.cursor.execute('SELECT id FROM ROW WHERE scan_id=%d and row_number=%d', (self.current_scan_id, row_number))
        row_id = self.cursor.fetchall()
        if row_id:
            if(self.debug): print(f"\tRow {row_number} already exists.")
            return row_id[0][0]
        else:
            if(self.debug): print(f"\tRow {row_number} doesn't exist")
            insertStatement = f"INSERT INTO ROW (scan_id, row_number) output Inserted.id VALUES ({self.current_scan_id},{row_number})"
            self.cursor.execute(insertStatement)
            row_id = self.cursor.fetchall()
            if(self.debug): print(f"\tCreated row {row_number}.")
            return row_id[0][0]
    
    def get_row_param(self, scan_name, param_name, conditions=''):
        '''
        Returns parameter value from ROW table associated with current scan.
        '''
        # Get id of the selected scan_name
        insertStatement = f'SELECT id FROM SKAN WHERE nazwa=\'{scan_name}\';'
        if(self.debug): print(insertStatement)
        self.cursor.execute(insertStatement)
        scan_id = self.cursor.fetchall()
        if scan_id:
            scan_id= scan_id[0][0]
        else:
            print(f"Scan with \"{scan_name}\" name doesn't exist. Can't get any parameters")
            exit()
            
        # Extend basic selection condition if there are any
        if len(conditions)>0: conditions = "and " + conditions 

        # Execute final condition
        insertStatement = 'SELECT {} FROM ROW WHERE scan_id={} {}'.format(param_name, scan_id, conditions)
        if(self.debug): print(insertStatement)
        self.cursor.execute(insertStatement)
        skan = self.cursor.fetchall()
        return skan

    def get_tooth_param(self, scan_name, param_name, row_number, conditions=''):
        '''
        Return value of the selected tooth parameters for the selected row
        '''
        row_id = self.get_row_param(scan_name,'id','row_number={}'.format(row_number))[0][0]
        if len(conditions)>0: conditions = "and " + conditions   
        insertStatement = f'SELECT {param_name} FROM TOOTH WHERE row_id={row_id} {conditions}'
        if(self.debug): print(insertStatement)
        self.cursor.execute(insertStatement)
        skan = self.cursor.fetchall()
        return skan

    def add_row_param(self, scan_name, stepienie, row_number):
        '''
        Update ROW table by adding cumulated 'stepienie' value for whole row
        '''
        # Get id of the selected scan_name
        insertStatement = f'SELECT id FROM SKAN WHERE nazwa=\'{scan_name}\''
        if(self.debug): print(insertStatement)
        self.cursor.execute(insertStatement)
        scan_id = self.cursor.fetchall()
        if scan_id:
            scan_id = scan_id[0][0]
        else:
            print(f"Scan with \"{scan_name}\" name doesn't exist. Can't get any parameters")
            exit()

        # Update table
        try:
            insertStatement = f"UPDATE ROW SET stepienie_row_value={stepienie} WHERE scan_id={scan_id} and row_number={row_number}"
            if(self.debug): print(insertStatement)
            self.cursor.execute(insertStatement)
            self.conn.commit()
        except:
            print('Can not upload declared row')

    def ovverride_row_stepienie(self, scan_name, stepienie, row_number):
        '''
        Update ROW table by adding custom 'stepienie' value for whole row
        '''
        # Get id of the selected scan_name
        insertStatement = f'SELECT id FROM SKAN WHERE nazwa=\'{scan_name}\''
        if(self.debug): print(insertStatement)
        self.cursor.execute(insertStatement)
        scan_id = self.cursor.fetchall()
        if scan_id:
            scan_id = scan_id[0][0]
        else:
            print(f"Scan with \"{scan_name}\" name doesn't exist. Can't get any parameters")
            exit()

        # Update table
        try:
            insertStatement = f"UPDATE ROW SET stepienie_correction={stepienie} WHERE scan_id={scan_id} and row_number={row_number}"
            if(self.debug): print(insertStatement)
            self.cursor.execute(insertStatement)
            self.conn.commit()
            return 1
        except:
            print('Can not upload declared row')
            return -1

    def select_from_view(self, scan_name, param_name, conditions=''):
        if len(conditions)>0: conditions = "and " + conditions 
        insertStatement = f"SELECT tooth_number, row_number, {param_name}, image_name FROM [View_1] WHERE nazwa=\'{scan_name}\' {conditions};"
        self.cursor.execute(insertStatement)
        data = self.cursor.fetchall()
        return data
    
    def get_table_names(self,table_name):
        insertStatement = f'SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = \'{table_name}\''
        self.cursor.execute(insertStatement)
        data = self.cursor.fetchall()
        return data
    
    def get_table_values(self,table_name):
        insertStatement = f'SELECT * FROM {table_name}'
        self.cursor.execute(insertStatement)
        data = self.cursor.fetchall()
        return data

    def update_broach_params(self,data):
        ids = [] # All ids from table
        for row in data:
            id = row['ID']
            ids.append(id)
            # Check if id is valid, if return error
            if(not id): 
                print('No ID in row')
                return -1
            # Find all columns names
            keys = []
            for key in row:
                if(key!='ID'):keys.append(key)
            # Fill columns with data
            params_u, params_c1, params_c2 = '','',''
            for i,key in enumerate(keys):
                v = str(row[key])
                # Formatting issues
                if v == 'True': v='1'
                if v == 'False': v='0'
                if v == 'None': v='none'
                if key == 'Project' or key == 'Verbao_id':
                    v = " \'{}\' ".format(v)
                # Creating insert Statement parameters
                if(i+1==len(keys)): 
                    params_u += str(key) + '=' + v
                    params_c1 += str(key)
                    params_c2 += v
                else: 
                    params_u += str(key) + '=' + v + ', '
                    params_c1 += str(key) + ', '
                    params_c2 += v + ', '
            
            # Check if id from table exist in database
            self.cursor.execute(f'SELECT ID FROM TypeOfBroach WHERE ID={id}')
            data = self.cursor.fetchall()
            data = np.array(data)
            data = np.reshape(data,(-1))
            if(data): base = 1 
            else: base = 0

            # Duplicated ids
            ids_set = set(ids)
            contains_duplicates = len(ids) != len(ids_set)
            if contains_duplicates == True:
                print("Can't add 2 broach types with the same id")
                return -1

            # Create record
            if(base == 0):
                # insertStatement = "UPDATE TypeOfBroach SET {} WHERE ID={}".format(params, id)
                insertStatement = f"INSERT INTO TypeOfBroach (ID, {params_c1}) VALUES ({id},{params_c2})"
                print('Create')

            # Update record
            if(base == 1):
                insertStatement = f"UPDATE TypeOfBroach SET {params_u} WHERE ID={id}"
                print('Update')

            # Execute insert statement
            try:
                if(self.debug): print(insertStatement)
                self.cursor.execute(insertStatement)
                self.conn.commit()
                print("Executed succesfuly")
            except:
                print('Can not upload declared row')
                return -1
        return 1

    def define_new_broach(self,data): 
        ids = [] # All ids from table
        for row in data:
            id = row['ID']
            ids.append(id)
            # Check if id is valid, if return error
            if(not id): 
                print('No ID in row')
                return -1
            # Find all columns names
            keys = []
            for key in row:
                if(key!='ID'):keys.append(key)
            # Fill columns with data
            params_u, params_c1, params_c2 = '','',''
            for i,key in enumerate(keys):
                v = str(row[key])
                # Formatting issues
                if v == 'True': v='1'
                if v == 'False': v='0'
                if key == 'DateOfEntry' or key == f'ScrappingDate':v = f" \'{v}\' "
                # Creating insert Statement parameters
                if(i+1==len(keys)): 
                    params_u += str(key) + '=' + v
                    params_c1 += str(key)
                    params_c2 += v
                else: 
                    params_u += str(key) + '=' + v + ', '
                    params_c1 += str(key) + ', '
                    params_c2 += v + ', '
            
            # Check if id from table exist in database
            self.cursor.execute(f'SELECT ID FROM Broach WHERE ID={id}')
            data = self.cursor.fetchall()
            data = np.array(data)
            data = np.reshape(data,(-1))
            if(data): base = 1 
            else: base = 0

            # Duplicated ids
            ids_set = set(ids)
            contains_duplicates = len(ids) != len(ids_set)
            if contains_duplicates == True:
                print("Can't add 2 broaches with the same id")
                return -1
           
            # Create record
            if(base == 0):
                # insertStatement = "UPDATE TypeOfBroach SET {} WHERE ID={}".format(params, id)
                insertStatement = f"INSERT INTO Broach (ID, {params_c1}) VALUES ({id},{params_c2})" 
                print('Create')

            # Update record
            if(base == 1):
                insertStatement = f"UPDATE Broach SET {params_u} WHERE ID={id}"
                print('Update')

            insertStatement = insertStatement.replace('\'None\'','null')
            # Execute insert statement
            try:
                if(self.debug): print(insertStatement)
                self.cursor.execute(insertStatement)
                self.conn.commit()
                print("Executed succesfuly")
            except:
                print('Can not upload declared row')
                return -1
                
        return 1