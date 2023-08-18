import React from 'react';
import { StyleSheet, Dimensions, Image, View, Text,TouchableOpacity,Linking} from 'react-native';
import { Actions } from 'react-native-router-flux';
import flags from './Images/flags.png'
import globe from './Images/globe.png'
import pointer from './Images/pointer.png'
import suit from './Images/suitcaseplus.png'
const LocationDetail = (props) => {
  const { location ,tlist,handleAddtList} = props;
  const joinJ = (id) =>{
    handleAddtList(id);
    alert("加入行程成功!");
    
  }
  return (
    
    <View>
      <Image style={styles.image} source={{ uri: location.properties.photo }} />
      <View style={styles.locationContent}>
        <Text style={styles.locationName}>{location.properties.name}</Text>
        <View style = {{justifyContent : 'space-between',flexDirection: 'row'}}>
          <View style={styles.inputWrap}>
            <Image Image style={styles.picon} source={pointer}/>
            <TouchableOpacity onPress = {() =>{
              Linking.openURL(location.properties.gmurl)
            }}>
              <Text style={styles.locationPrice}>導航</Text>
            </TouchableOpacity>
          </View>

          <View>
            <TouchableOpacity style={styles.buttonWrap} onPress = {() =>{
              joinJ(location.id);
            }}>
                <Image Image style={styles.icon} source={suit}/>
                <Text style={styles.locationPrice}>加入行程</Text>
              </TouchableOpacity>
                
          </View>
        </View>

        <View style={styles.addressWrap}>
          <Image Image style={styles.icon} source={flags}/>
          <Text style={styles.locationPrice}></Text>
          <Text style={styles.address}> {location.properties.address}</Text>
        </View>
        <View style={styles.addressWrap}>
          <Image Image style={styles.icon} source={globe}/>
          <TouchableOpacity onPress = {() =>{
              if (location.properties.website != "null")
                Linking.openURL(location.properties.website)
              else
                alert("No url Found!")
            }}>
              <Text style={styles.address}>{location.properties.website}</Text>
              
            </TouchableOpacity>
          
        </View>

      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  locationList: {
    flex: 1,
  },
  image: {
    width: Dimensions.get('window').width,
    height: Dimensions.get('window').width / 2,
  },
  locationContent: {
    marginLeft: 10,
  },
  locationName: {
    fontSize: 24,
    fontWeight: 'bold',
    paddingVertical: 3,
  },
  locationPrice: {
    fontSize: 18,
    fontWeight: 'bold',
    color: 'black',
    paddingBottom: 10,
  },
  locationDesc: {
    fontSize: 14,
    color: 'gray',
  }, 
  icon: {
    width: 24,
    height: 24,
    marginRight:10
  },
  picon: {
    width: 30,
    height: 30,
    marginRight:4
  },
  inputWrap:{
    flex:0,
    flexDirection:'row',
    alignItems:'center',
    width:245,
    height:45,

    borderColor:'rgba(171, 190, 215, 0.56)',

    marginBottom:0,
    },
    buttonWrap:{
      flex:0,
      flexDirection:'row',
      alignItems:'center',
      width:300,
      height:45,
      borderColor:'rgba(171, 190, 215, 0.56)',
      marginBottom:0,
      },
      address: {
        fontSize: 17,
        fontWeight: 'bold',
        color: 'black',
        paddingBottom: 10,
      },
      addressWrap:{
        flex:0,
        flexDirection:'row',
        alignItems:'center',
        width:375,
        height:45,
    
        borderColor:'rgba(171, 190, 215, 0.56)',
    
        marginBottom:0,
        },
});

export default LocationDetail;
