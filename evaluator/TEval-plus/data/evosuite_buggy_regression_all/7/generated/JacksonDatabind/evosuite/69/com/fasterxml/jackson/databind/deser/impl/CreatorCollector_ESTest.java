/*
 * This file was automatically generated by EvoSuite
 * Sat Jul 29 19:06:49 GMT 2023
 */

package com.fasterxml.jackson.databind.deser.impl;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.ObjectCodec;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.MapperFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.cfg.DeserializerFactoryConfig;
import com.fasterxml.jackson.databind.deser.BeanDeserializerFactory;
import com.fasterxml.jackson.databind.deser.DefaultDeserializationContext;
import com.fasterxml.jackson.databind.deser.impl.CreatorCollector;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import java.io.IOException;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class CreatorCollector_ESTest extends CreatorCollector_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      JsonFactory jsonFactory0 = new JsonFactory();
      ObjectMapper objectMapper0 = new ObjectMapper(jsonFactory0);
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory(false);
      ArrayNode arrayNode0 = new ArrayNode(jsonNodeFactory0);
      JsonParser jsonParser0 = arrayNode0.traverse((ObjectCodec) objectMapper0);
      Class<Method> class0 = Method.class;
      try { 
        objectMapper0.readValue(jsonParser0, class0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Can not deserialize instance of java.lang.reflect.Method out of START_ARRAY token
         //  at [Source: java.lang.String@0000000002; line: -1, column: -1]
         //
         verifyException("com.fasterxml.jackson.databind.JsonMappingException", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      JsonFactory jsonFactory0 = new JsonFactory();
      ObjectMapper objectMapper0 = new ObjectMapper(jsonFactory0);
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory(false);
      MapperFeature[] mapperFeatureArray0 = new MapperFeature[5];
      MapperFeature mapperFeature0 = MapperFeature.IGNORE_DUPLICATE_MODULE_REGISTRATIONS;
      mapperFeatureArray0[0] = mapperFeature0;
      MapperFeature mapperFeature1 = MapperFeature.AUTO_DETECT_CREATORS;
      mapperFeatureArray0[1] = mapperFeature1;
      mapperFeatureArray0[2] = mapperFeatureArray0[0];
      mapperFeatureArray0[3] = mapperFeatureArray0[1];
      mapperFeatureArray0[4] = mapperFeature1;
      objectMapper0.disable(mapperFeatureArray0);
      ArrayNode arrayNode0 = new ArrayNode(jsonNodeFactory0);
      JsonParser jsonParser0 = arrayNode0.traverse((ObjectCodec) objectMapper0);
      Class<Method> class0 = Method.class;
      try { 
        objectMapper0.readValue(jsonParser0, class0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Can not deserialize instance of java.lang.reflect.Method out of START_ARRAY token
         //  at [Source: java.lang.String@0000000002; line: -1, column: -1]
         //
         verifyException("com.fasterxml.jackson.databind.JsonMappingException", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      CreatorCollector.Vanilla creatorCollector_Vanilla0 = new CreatorCollector.Vanilla(2015);
      boolean boolean0 = creatorCollector_Vanilla0.canInstantiate();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      CreatorCollector.Vanilla creatorCollector_Vanilla0 = new CreatorCollector.Vanilla(1);
      String string0 = creatorCollector_Vanilla0.getValueTypeDesc();
      assertEquals("java.util.ArrayList", string0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      CreatorCollector.Vanilla creatorCollector_Vanilla0 = new CreatorCollector.Vanilla(2);
      String string0 = creatorCollector_Vanilla0.getValueTypeDesc();
      assertEquals("java.util.LinkedHashMap", string0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      CreatorCollector.Vanilla creatorCollector_Vanilla0 = new CreatorCollector.Vanilla(3);
      String string0 = creatorCollector_Vanilla0.getValueTypeDesc();
      assertEquals("java.util.HashMap", string0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      CreatorCollector.Vanilla creatorCollector_Vanilla0 = new CreatorCollector.Vanilla((-1607));
      String string0 = creatorCollector_Vanilla0.getValueTypeDesc();
      assertEquals("java.lang.Object", string0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      CreatorCollector.Vanilla creatorCollector_Vanilla0 = new CreatorCollector.Vanilla(1);
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory(deserializerFactoryConfig0);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      ArrayList arrayList0 = (ArrayList)creatorCollector_Vanilla0.createUsingDefault(defaultDeserializationContext_Impl0);
      assertTrue(arrayList0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      CreatorCollector.Vanilla creatorCollector_Vanilla0 = new CreatorCollector.Vanilla(2);
      JsonFactory jsonFactory0 = new JsonFactory();
      ObjectMapper objectMapper0 = new ObjectMapper(jsonFactory0);
      DeserializationContext deserializationContext0 = objectMapper0.getDeserializationContext();
      LinkedHashMap linkedHashMap0 = (LinkedHashMap)creatorCollector_Vanilla0.createUsingDefault(deserializationContext0);
      assertEquals(0, linkedHashMap0.size());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      CreatorCollector.Vanilla creatorCollector_Vanilla0 = new CreatorCollector.Vanilla(3);
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory(deserializerFactoryConfig0);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      HashMap hashMap0 = (HashMap)creatorCollector_Vanilla0.createUsingDefault(defaultDeserializationContext_Impl0);
      assertEquals(0, hashMap0.size());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      CreatorCollector.Vanilla creatorCollector_Vanilla0 = new CreatorCollector.Vanilla(2015);
      // Undeclared exception!
      try { 
        creatorCollector_Vanilla0.createUsingDefault((DeserializationContext) null);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // Unknown type 2015
         //
         verifyException("com.fasterxml.jackson.databind.deser.impl.CreatorCollector$Vanilla", e);
      }
  }
}