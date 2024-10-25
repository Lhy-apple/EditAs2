/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 21:48:38 GMT 2023
 */

package com.fasterxml.jackson.databind.deser;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.annotation.ObjectIdGenerator;
import com.fasterxml.jackson.annotation.SimpleObjectIdResolver;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JsonDeserializer;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
import com.fasterxml.jackson.databind.PropertyName;
import com.fasterxml.jackson.databind.deser.BasicDeserializerFactory;
import com.fasterxml.jackson.databind.deser.BeanDeserializer;
import com.fasterxml.jackson.databind.deser.BeanDeserializerBase;
import com.fasterxml.jackson.databind.deser.SettableBeanProperty;
import com.fasterxml.jackson.databind.deser.UnresolvedForwardReference;
import com.fasterxml.jackson.databind.deser.impl.BeanPropertyMap;
import com.fasterxml.jackson.databind.deser.impl.ObjectIdReader;
import com.fasterxml.jackson.databind.deser.impl.PropertyValueBuffer;
import com.fasterxml.jackson.databind.type.MapType;
import com.fasterxml.jackson.databind.type.TypeFactory;
import com.fasterxml.jackson.databind.util.NameTransformer;
import java.time.ZoneId;
import java.time.format.TextStyle;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class BeanDeserializer_ESTest extends BeanDeserializer_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      BeanDeserializer beanDeserializer0 = null;
      try {
        beanDeserializer0 = new BeanDeserializer((BeanDeserializerBase) null, false);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.BeanDeserializerBase", e);
      }
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      BeanDeserializer beanDeserializer0 = null;
      try {
        beanDeserializer0 = new BeanDeserializer((BeanDeserializerBase) null, (NameTransformer) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.BeanDeserializerBase", e);
      }
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      TreeSet<SettableBeanProperty> treeSet0 = new TreeSet<SettableBeanProperty>();
      LinkedHashMap<String, List<PropertyName>> linkedHashMap0 = new LinkedHashMap<String, List<PropertyName>>();
      BeanPropertyMap beanPropertyMap0 = new BeanPropertyMap(true, treeSet0, linkedHashMap0);
      BeanDeserializer beanDeserializer0 = null;
      try {
        beanDeserializer0 = new BeanDeserializer((BeanDeserializerBase) null, beanPropertyMap0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.BeanDeserializerBase", e);
      }
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      BeanDeserializer beanDeserializer0 = null;
      try {
        beanDeserializer0 = new BeanDeserializer((BeanDeserializerBase) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.BeanDeserializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      Set<String> set0 = ZoneId.getAvailableZoneIds();
      BeanDeserializer beanDeserializer0 = null;
      try {
        beanDeserializer0 = new BeanDeserializer((BeanDeserializerBase) null, set0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.BeanDeserializerBase", e);
      }
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      JsonFactory jsonFactory0 = new JsonFactory();
      ObjectMapper objectMapper0 = new ObjectMapper(jsonFactory0);
      SimpleObjectIdResolver simpleObjectIdResolver0 = new SimpleObjectIdResolver();
      ObjectReader objectReader0 = objectMapper0.readerForUpdating(simpleObjectIdResolver0);
      TypeFactory typeFactory0 = objectReader0.getTypeFactory();
      Class<HashMap> class0 = HashMap.class;
      MapType mapType0 = typeFactory0.constructRawMapType(class0);
      PropertyName propertyName0 = BasicDeserializerFactory.UNWRAPPED_CREATOR_PARAM_NAME;
      ObjectIdGenerator<Object> objectIdGenerator0 = (ObjectIdGenerator<Object>) mock(ObjectIdGenerator.class, new ViolatedAssumptionAnswer());
      ObjectIdReader objectIdReader0 = ObjectIdReader.construct(mapType0, propertyName0, objectIdGenerator0, (JsonDeserializer<?>) null, (SettableBeanProperty) null, simpleObjectIdResolver0);
      BeanDeserializer beanDeserializer0 = null;
      try {
        beanDeserializer0 = new BeanDeserializer((BeanDeserializerBase) null, objectIdReader0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.BeanDeserializerBase", e);
      }
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      JsonFactory jsonFactory0 = new JsonFactory();
      byte[] byteArray0 = new byte[3];
      JsonParser jsonParser0 = jsonFactory0.createParser(byteArray0, (int) (byte)53, (-65281));
      UnresolvedForwardReference unresolvedForwardReference0 = new UnresolvedForwardReference(jsonParser0, "JSON");
      ObjectMapper objectMapper0 = new ObjectMapper(jsonFactory0);
      SimpleObjectIdResolver simpleObjectIdResolver0 = new SimpleObjectIdResolver();
      ObjectReader objectReader0 = objectMapper0.readerForUpdating(simpleObjectIdResolver0);
      TypeFactory typeFactory0 = objectReader0.getTypeFactory();
      Class<HashMap> class0 = HashMap.class;
      MapType mapType0 = typeFactory0.constructRawMapType(class0);
      BeanDeserializer.BeanReferring beanDeserializer_BeanReferring0 = new BeanDeserializer.BeanReferring((DeserializationContext) null, unresolvedForwardReference0, mapType0, (PropertyValueBuffer) null, (SettableBeanProperty) null);
      TextStyle textStyle0 = TextStyle.SHORT_STANDALONE;
      beanDeserializer_BeanReferring0.setBean(jsonFactory0);
      TextStyle textStyle1 = textStyle0.asStandalone();
      // Undeclared exception!
      try { 
        beanDeserializer_BeanReferring0.handleResolvedForwardReference(jsonFactory0, textStyle1);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.BeanDeserializer$BeanReferring", e);
      }
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      JsonFactory jsonFactory0 = new JsonFactory();
      byte[] byteArray0 = new byte[3];
      JsonParser jsonParser0 = jsonFactory0.createParser(byteArray0, (int) (byte)53, (-65281));
      UnresolvedForwardReference unresolvedForwardReference0 = new UnresolvedForwardReference(jsonParser0, "JSON");
      ObjectMapper objectMapper0 = new ObjectMapper(jsonFactory0);
      SimpleObjectIdResolver simpleObjectIdResolver0 = new SimpleObjectIdResolver();
      ObjectReader objectReader0 = objectMapper0.readerForUpdating(simpleObjectIdResolver0);
      TypeFactory typeFactory0 = objectReader0.getTypeFactory();
      Class<HashMap> class0 = HashMap.class;
      MapType mapType0 = typeFactory0.constructRawMapType(class0);
      BeanDeserializer.BeanReferring beanDeserializer_BeanReferring0 = new BeanDeserializer.BeanReferring((DeserializationContext) null, unresolvedForwardReference0, mapType0, (PropertyValueBuffer) null, (SettableBeanProperty) null);
      TextStyle textStyle0 = TextStyle.SHORT_STANDALONE;
      TextStyle textStyle1 = textStyle0.asStandalone();
      // Undeclared exception!
      try { 
        beanDeserializer_BeanReferring0.handleResolvedForwardReference(jsonFactory0, textStyle1);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.BeanDeserializer$BeanReferring", e);
      }
  }
}
