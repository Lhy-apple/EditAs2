/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 13:42:23 GMT 2023
 */

package com.fasterxml.jackson.databind.deser;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.annotation.JsonTypeInfo;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.type.ResolvedType;
import com.fasterxml.jackson.databind.BeanDescription;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.Module;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
import com.fasterxml.jackson.databind.cfg.DeserializerFactoryConfig;
import com.fasterxml.jackson.databind.deser.BeanDeserializer;
import com.fasterxml.jackson.databind.deser.BeanDeserializerFactory;
import com.fasterxml.jackson.databind.deser.DefaultDeserializationContext;
import com.fasterxml.jackson.databind.deser.DeserializerCache;
import com.fasterxml.jackson.databind.node.DoubleNode;
import com.fasterxml.jackson.databind.node.MissingNode;
import com.fasterxml.jackson.databind.type.CollectionLikeType;
import com.fasterxml.jackson.databind.type.MapLikeType;
import com.fasterxml.jackson.databind.type.PlaceholderForType;
import com.fasterxml.jackson.databind.type.TypeFactory;
import java.io.IOException;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class DeserializerCache_ESTest extends DeserializerCache_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      DeserializerCache deserializerCache0 = new DeserializerCache();
      int int0 = deserializerCache0.cachedDeserializersCount();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      DeserializerCache deserializerCache0 = new DeserializerCache();
      Object object0 = deserializerCache0.writeReplace();
      assertSame(deserializerCache0, object0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      DeserializerCache deserializerCache0 = new DeserializerCache();
      deserializerCache0.flushCachedDeserializers();
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      DeserializerCache deserializerCache0 = new DeserializerCache();
      // Undeclared exception!
      try { 
        deserializerCache0._handleUnknownKeyDeserializer((DeserializationContext) null, javaType0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.DeserializerCache", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      PlaceholderForType placeholderForType0 = new PlaceholderForType(1957);
      objectMapper0.readValues((JsonParser) null, (ResolvedType) placeholderForType0);
      Class<Object> class0 = Object.class;
      ObjectReader objectReader0 = objectMapper0.readerFor(class0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      DeserializerCache deserializerCache0 = new DeserializerCache();
      // Undeclared exception!
      try { 
        deserializerCache0._findCachedDeserializer((JavaType) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Null JavaType passed
         //
         verifyException("com.fasterxml.jackson.databind.deser.DeserializerCache", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      objectMapper0.enableDefaultTyping();
      Class<BeanDeserializer> class0 = BeanDeserializer.class;
      ObjectReader objectReader0 = objectMapper0.readerFor(class0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<DoubleNode> class0 = DoubleNode.class;
      MapLikeType mapLikeType0 = typeFactory0.constructMapLikeType(class0, class0, class0);
      // Undeclared exception!
      try { 
        defaultDeserializationContext_Impl0.findNonContextualValueDeserializer(mapLikeType0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.DeserializerCache", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper((JsonFactory) null);
      Class<JsonTypeInfo.As> class0 = JsonTypeInfo.As.class;
      ObjectReader objectReader0 = objectMapper0.readerFor(class0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      DeserializerCache deserializerCache0 = new DeserializerCache();
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<Module> class0 = Module.class;
      MapLikeType mapLikeType0 = typeFactory0.constructMapLikeType(class0, class0, class0);
      // Undeclared exception!
      try { 
        deserializerCache0._createDeserializer2(defaultDeserializationContext_Impl0, beanDeserializerFactory0, mapLikeType0, (BeanDescription) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.BasicDeserializerFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      Class<DoubleNode> class0 = DoubleNode.class;
      ObjectReader objectReader0 = objectMapper0.readerFor(class0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      DeserializerCache deserializerCache0 = new DeserializerCache();
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory(deserializerFactoryConfig0);
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<CollectionLikeType> class0 = CollectionLikeType.class;
      MapLikeType mapLikeType0 = typeFactory0.constructMapLikeType(class0, (JavaType) null, (JavaType) null);
      // Undeclared exception!
      try { 
        deserializerCache0._createAndCacheValueDeserializer((DeserializationContext) null, beanDeserializerFactory0, mapLikeType0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.DeserializerCache", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      DeserializerCache deserializerCache0 = new DeserializerCache();
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory(deserializerFactoryConfig0);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<JsonTypeInfo.As> class0 = JsonTypeInfo.As.class;
      CollectionLikeType collectionLikeType0 = typeFactory0.constructRawCollectionLikeType(class0);
      MissingNode missingNode0 = MissingNode.getInstance();
      CollectionLikeType collectionLikeType1 = collectionLikeType0.withContentValueHandler(missingNode0);
      // Undeclared exception!
      try { 
        deserializerCache0.hasValueDeserializerFor(defaultDeserializationContext_Impl0, beanDeserializerFactory0, collectionLikeType1);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.DeserializerCache", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      DeserializerCache deserializerCache0 = new DeserializerCache();
      // Undeclared exception!
      try { 
        deserializerCache0._handleUnknownValueDeserializer((DeserializationContext) null, javaType0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.DeserializerCache", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<Module> class0 = Module.class;
      MapLikeType mapLikeType0 = typeFactory0.constructMapLikeType(class0, class0, class0);
      CollectionLikeType collectionLikeType0 = CollectionLikeType.upgradeFrom(mapLikeType0, mapLikeType0);
      DeserializerCache deserializerCache0 = new DeserializerCache();
      try { 
        deserializerCache0._handleUnknownValueDeserializer(defaultDeserializationContext_Impl0, collectionLikeType0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Can not find a Value deserializer for abstract type [collection-like type; class com.fasterxml.jackson.databind.Module, contains [map-like type; class com.fasterxml.jackson.databind.Module, [simple type, class com.fasterxml.jackson.databind.Module] -> [simple type, class com.fasterxml.jackson.databind.Module]]]
         //
         verifyException("com.fasterxml.jackson.databind.JsonMappingException", e);
      }
  }
}