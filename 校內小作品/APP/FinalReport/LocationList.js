import React from 'react';
import { StyleSheet, ScrollView ,TextInput,View,Image,FlatList} from 'react-native';
import { Actions } from 'react-native-router-flux';
import SearchInput, { createFilter } from 'react-native-search-filter';
import LocationItem from './LocationItem';
import search from './Images/search.png'
import loJson from './point.json'
const KEYS_TO_FILTERS = ['properties.name'];
class LoactionList extends React.Component {
  constructor(props) {
    super(props);

    this.state = {
      tlist : props.tlists,
      locations: loJson.features,
      searchString : "",
      loading: false,            
      error: null,  
    };

  }
  searchUpdated(term) {
    this.setState({ searchString: term })
  }
  handleRedirectLocationDetail = (id) => {
    const { locations, tlist } = this.state;
    const {handleAddtList} = this.props;
    const location = locations.find((location) => location.id === id);

    // 跳轉至餐點詳細頁面時將底部的 Tab 隱藏
    Actions.push('LocationDetail', { location: location, hideTabBar: true ,tlist : tlist,handleAddtList:handleAddtList});
  };

  render() {
    const filteredPoint = this.state.locations.filter(createFilter(this.state.searchString,KEYS_TO_FILTERS))

    return (
      <View style={styles.locationList}>
          <View style={styles.inputWrap}>
            <Image Image style={styles.icon} source={search}/>
            {/* {alert(this.state.tlist[0].id)} */}
            <SearchInput
                style={styles.textInput}
                placeholder="Search Something!"
                onChangeText={(term) => {this.searchUpdated(term)}}
                underlineColorAndroid="transparent"
            />
            
          </View>
 
          {/* <ScrollView >
            {this.state.locations.map((location) => {
              return <LocationItem key={location.id} location={location} onPress={this.handleRedirectLocationDetail} />;
            })}
          </ScrollView>
           */}
          <FlatList
            data = {filteredPoint}
            renderItem  = {({item}) => <LocationItem key={item.id} location={item} onPress={this.handleRedirectLocationDetail} />}
            keyExtractor={item => item.id.toString()} 
            initialNumToRender={6}
          />



      </View>

    );
  }
}

const styles = StyleSheet.create({
  locationList: {
    flex: 1,
  },
  locationItem: {
    flexDirection: 'row',
    alignItems: 'center',
    borderBottomColor: '#DDD',
    borderBottomWidth: 1,
    paddingVertical: 10,
  },
  image: {
    width: 100,
    height: 100,
  },
  locationContent: {
    marginLeft: 10,
  },
  locationName: {
    fontSize: 14,
    fontWeight: 'bold',
    textAlign: 'center',
    paddingVertical: 3,
  },
  locationPrice: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#F00',
  },
  searchIcon: {
    padding: 10,
  },
  input: {
    flex: 0,
    paddingTop: 10,
    paddingRight: 10,
    paddingBottom: 10,
    paddingLeft: 0,
    backgroundColor: '#fff',
    color: '#424242',
  },
  inputWrap:{
    flex:0,
    flexDirection:'row',
    alignItems:'center',
    width:245,
    height:45,
    backgroundColor:'transparent',
    borderColor:'rgba(171, 190, 215, 0.56)',
    borderBottomWidth: 1,
    marginBottom:0,
    },
  icon: {
    width: 16,
    height: 16,
    marginRight:10
  },
  textInput:{
    backgroundColor:'transparent',
    borderColor:'transparent',
    borderWidth: 1,
    width:200,
    height:50,
    fontSize:14,
    color:'black',
  },
});

export default LoactionList;
