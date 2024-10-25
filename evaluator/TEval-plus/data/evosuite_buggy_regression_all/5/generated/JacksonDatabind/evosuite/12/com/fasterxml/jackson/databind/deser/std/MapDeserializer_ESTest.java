/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 23:00:12 GMT 2023
 */

package com.fasterxml.jackson.databind.deser.std;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.annotation.ObjectIdGenerators;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.JsonDeserializer;
import com.fasterxml.jackson.databind.JsonMappingException;
import com.fasterxml.jackson.databind.KeyDeserializer;
import com.fasterxml.jackson.databind.Module;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
import com.fasterxml.jackson.databind.cfg.DeserializerFactoryConfig;
import com.fasterxml.jackson.databind.deser.BeanDeserializerFactory;
import com.fasterxml.jackson.databind.deser.DefaultDeserializationContext;
import com.fasterxml.jackson.databind.deser.SettableBeanProperty;
import com.fasterxml.jackson.databind.deser.impl.PropertyBasedCreator;
import com.fasterxml.jackson.databind.deser.std.FromStringDeserializer;
import com.fasterxml.jackson.databind.deser.std.JsonLocationInstantiator;
import com.fasterxml.jackson.databind.deser.std.MapDeserializer;
import com.fasterxml.jackson.databind.deser.std.StdKeyDeserializer;
import com.fasterxml.jackson.databind.jsontype.TypeDeserializer;
import com.fasterxml.jackson.databind.jsontype.impl.AsExternalTypeDeserializer;
import com.fasterxml.jackson.databind.jsontype.impl.AsWrapperTypeDeserializer;
import com.fasterxml.jackson.databind.jsontype.impl.ClassNameIdResolver;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import com.fasterxml.jackson.databind.type.ClassKey;
import com.fasterxml.jackson.databind.type.MapLikeType;
import com.fasterxml.jackson.databind.type.MapType;
import com.fasterxml.jackson.databind.type.SimpleType;
import com.fasterxml.jackson.databind.type.TypeBindings;
import com.fasterxml.jackson.databind.type.TypeFactory;
import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Type;
import java.sql.SQLIntegrityConstraintViolationException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.Map;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.io.MockIOException;
import org.evosuite.runtime.mock.java.lang.MockError;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class MapDeserializer_ESTest extends MapDeserializer_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      SimpleType simpleType0 = (SimpleType)TypeBindings.UNBOUND;
      JsonLocationInstantiator jsonLocationInstantiator0 = new JsonLocationInstantiator();
      MapDeserializer mapDeserializer0 = new MapDeserializer(simpleType0, jsonLocationInstantiator0, (KeyDeserializer) null, (JsonDeserializer<Object>) null, (TypeDeserializer) null);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      // Undeclared exception!
      try { 
        mapDeserializer0.deserializeWithType((JsonParser) null, defaultDeserializationContext_Impl0, (TypeDeserializer) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.MapDeserializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      SimpleType simpleType0 = (SimpleType)TypeBindings.UNBOUND;
      JsonLocationInstantiator jsonLocationInstantiator0 = new JsonLocationInstantiator();
      MapDeserializer mapDeserializer0 = new MapDeserializer(simpleType0, jsonLocationInstantiator0, (KeyDeserializer) null, (JsonDeserializer<Object>) null, (TypeDeserializer) null);
      MapDeserializer mapDeserializer1 = new MapDeserializer(mapDeserializer0);
      assertTrue(mapDeserializer1.isCachable());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      SimpleType simpleType0 = (SimpleType)TypeBindings.UNBOUND;
      JsonLocationInstantiator jsonLocationInstantiator0 = new JsonLocationInstantiator();
      MapDeserializer mapDeserializer0 = new MapDeserializer(simpleType0, jsonLocationInstantiator0, (KeyDeserializer) null, (JsonDeserializer<Object>) null, (TypeDeserializer) null);
      JavaType javaType0 = mapDeserializer0.getValueType();
      assertFalse(javaType0.isArrayType());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      SimpleType simpleType0 = (SimpleType)TypeBindings.UNBOUND;
      JsonLocationInstantiator jsonLocationInstantiator0 = new JsonLocationInstantiator();
      SQLIntegrityConstraintViolationException sQLIntegrityConstraintViolationException0 = new SQLIntegrityConstraintViolationException("");
      InvocationTargetException invocationTargetException0 = new InvocationTargetException(sQLIntegrityConstraintViolationException0, "");
      MapDeserializer mapDeserializer0 = new MapDeserializer(simpleType0, jsonLocationInstantiator0, (KeyDeserializer) null, (JsonDeserializer<Object>) null, (TypeDeserializer) null);
      JsonMappingException jsonMappingException0 = JsonMappingException.wrapWithPath((Throwable) invocationTargetException0, (Object) simpleType0, "");
      // Undeclared exception!
      try { 
        mapDeserializer0.wrapAndThrow(jsonMappingException0, (Object) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // Can not pass null fieldName
         //
         verifyException("com.fasterxml.jackson.databind.JsonMappingException$Reference", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      SimpleType simpleType0 = (SimpleType)TypeBindings.UNBOUND;
      JsonLocationInstantiator jsonLocationInstantiator0 = new JsonLocationInstantiator();
      MapDeserializer mapDeserializer0 = new MapDeserializer(simpleType0, jsonLocationInstantiator0, (KeyDeserializer) null, (JsonDeserializer<Object>) null, (TypeDeserializer) null);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      MapDeserializer mapDeserializer1 = mapDeserializer0.withResolved((KeyDeserializer) null, (TypeDeserializer) null, mapDeserializer0, linkedHashSet0);
      MapDeserializer mapDeserializer2 = mapDeserializer1.withResolved((KeyDeserializer) null, (TypeDeserializer) null, mapDeserializer0, linkedHashSet0);
      assertSame(mapDeserializer2, mapDeserializer1);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      SimpleType simpleType0 = (SimpleType)TypeBindings.UNBOUND;
      JsonLocationInstantiator jsonLocationInstantiator0 = new JsonLocationInstantiator();
      MapDeserializer mapDeserializer0 = new MapDeserializer(simpleType0, jsonLocationInstantiator0, (KeyDeserializer) null, (JsonDeserializer<Object>) null, (TypeDeserializer) null);
      // Undeclared exception!
      try { 
        mapDeserializer0.findBackReference("");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Can not handle managed/back reference '': type: container deserializer of type com.fasterxml.jackson.databind.deser.std.MapDeserializer returned null for 'getContentDeserializer()'
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.ContainerDeserializerBase", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      SimpleType simpleType0 = (SimpleType)TypeBindings.UNBOUND;
      JsonLocationInstantiator jsonLocationInstantiator0 = new JsonLocationInstantiator();
      MapDeserializer mapDeserializer0 = new MapDeserializer(simpleType0, jsonLocationInstantiator0, (KeyDeserializer) null, (JsonDeserializer<Object>) null, (TypeDeserializer) null);
      // Undeclared exception!
      try { 
        mapDeserializer0.deserialize((JsonParser) null, (DeserializationContext) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.MapDeserializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      SimpleType simpleType0 = (SimpleType)TypeBindings.UNBOUND;
      JsonLocationInstantiator jsonLocationInstantiator0 = new JsonLocationInstantiator();
      MapDeserializer mapDeserializer0 = new MapDeserializer(simpleType0, jsonLocationInstantiator0, (KeyDeserializer) null, (JsonDeserializer<Object>) null, (TypeDeserializer) null);
      JavaType javaType0 = mapDeserializer0.getContentType();
      assertNull(javaType0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      SimpleType simpleType0 = (SimpleType)TypeBindings.UNBOUND;
      ObjectMapper objectMapper0 = new ObjectMapper();
      ObjectReader objectReader0 = objectMapper0.reader();
      ObjectReader objectReader1 = objectReader0.withType((Type) simpleType0);
      assertNotSame(objectReader1, objectReader0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      SimpleType simpleType0 = (SimpleType)TypeBindings.UNBOUND;
      JsonLocationInstantiator jsonLocationInstantiator0 = new JsonLocationInstantiator();
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      ClassNameIdResolver classNameIdResolver0 = new ClassNameIdResolver(simpleType0, typeFactory0);
      Class<Object> class0 = Object.class;
      AsWrapperTypeDeserializer asWrapperTypeDeserializer0 = new AsWrapperTypeDeserializer(simpleType0, classNameIdResolver0, "6Jj+v7)FlpD6h", true, class0);
      MapDeserializer mapDeserializer0 = new MapDeserializer(simpleType0, jsonLocationInstantiator0, (KeyDeserializer) null, (JsonDeserializer<Object>) null, asWrapperTypeDeserializer0);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      MapDeserializer mapDeserializer1 = mapDeserializer0.withResolved((KeyDeserializer) null, (TypeDeserializer) null, (JsonDeserializer<?>) null, linkedHashSet0);
      assertFalse(mapDeserializer1.isCachable());
      assertFalse(mapDeserializer0.isCachable());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      SimpleType simpleType0 = (SimpleType)TypeBindings.UNBOUND;
      JsonLocationInstantiator jsonLocationInstantiator0 = new JsonLocationInstantiator();
      MapDeserializer mapDeserializer0 = new MapDeserializer(simpleType0, jsonLocationInstantiator0, (KeyDeserializer) null, (JsonDeserializer<Object>) null, (TypeDeserializer) null);
      HashSet<String> hashSet0 = new HashSet<String>();
      MapDeserializer mapDeserializer1 = mapDeserializer0.withResolved((KeyDeserializer) null, (TypeDeserializer) null, (JsonDeserializer<?>) null, hashSet0);
      assertFalse(mapDeserializer1.isCachable());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      SimpleType simpleType0 = (SimpleType)TypeBindings.UNBOUND;
      JsonLocationInstantiator jsonLocationInstantiator0 = new JsonLocationInstantiator();
      Class<String> class0 = String.class;
      StdKeyDeserializer.StringKD stdKeyDeserializer_StringKD0 = StdKeyDeserializer.StringKD.forType(class0);
      MapDeserializer mapDeserializer0 = new MapDeserializer(simpleType0, jsonLocationInstantiator0, stdKeyDeserializer_StringKD0, (JsonDeserializer<Object>) null, (TypeDeserializer) null);
      assertTrue(mapDeserializer0.isCachable());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Class<Module> class0 = Module.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      MapLikeType mapLikeType0 = MapLikeType.construct(class0, simpleType0, simpleType0);
      MapType mapType0 = MapType.construct(class0, mapLikeType0, simpleType0);
      JsonLocationInstantiator jsonLocationInstantiator0 = new JsonLocationInstantiator();
      Class<ObjectIdGenerators.IntSequenceGenerator> class1 = ObjectIdGenerators.IntSequenceGenerator.class;
      StdKeyDeserializer.StringKD stdKeyDeserializer_StringKD0 = StdKeyDeserializer.StringKD.forType(class1);
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class1, 2);
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      ClassNameIdResolver classNameIdResolver0 = new ClassNameIdResolver(mapLikeType0, typeFactory0);
      Class<InputStream> class2 = InputStream.class;
      AsExternalTypeDeserializer asExternalTypeDeserializer0 = new AsExternalTypeDeserializer(mapType0, classNameIdResolver0, "!e9O^!9 {g_etb|%H)", false, class2);
      MapDeserializer mapDeserializer0 = new MapDeserializer(mapType0, jsonLocationInstantiator0, stdKeyDeserializer_StringKD0, fromStringDeserializer_Std0, asExternalTypeDeserializer0);
      assertFalse(mapDeserializer0.isCachable());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      SimpleType simpleType0 = (SimpleType)TypeBindings.UNBOUND;
      Class<ClassKey> class0 = ClassKey.class;
      MapLikeType mapLikeType0 = MapLikeType.construct(class0, simpleType0, simpleType0);
      JsonLocationInstantiator jsonLocationInstantiator0 = new JsonLocationInstantiator();
      Class<String> class1 = String.class;
      StdKeyDeserializer.StringKD stdKeyDeserializer_StringKD0 = StdKeyDeserializer.StringKD.forType(class1);
      MapDeserializer mapDeserializer0 = new MapDeserializer(mapLikeType0, jsonLocationInstantiator0, stdKeyDeserializer_StringKD0, (JsonDeserializer<Object>) null, (TypeDeserializer) null);
      assertTrue(mapDeserializer0.isCachable());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      SimpleType simpleType0 = (SimpleType)TypeBindings.UNBOUND;
      JsonLocationInstantiator jsonLocationInstantiator0 = new JsonLocationInstantiator();
      MapDeserializer mapDeserializer0 = new MapDeserializer(simpleType0, jsonLocationInstantiator0, (KeyDeserializer) null, (JsonDeserializer<Object>) null, (TypeDeserializer) null);
      String[] stringArray0 = new String[5];
      mapDeserializer0.setIgnorableProperties(stringArray0);
      assertFalse(mapDeserializer0.isCachable());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      SimpleType simpleType0 = (SimpleType)TypeBindings.UNBOUND;
      JsonLocationInstantiator jsonLocationInstantiator0 = new JsonLocationInstantiator();
      MapDeserializer mapDeserializer0 = new MapDeserializer(simpleType0, jsonLocationInstantiator0, (KeyDeserializer) null, (JsonDeserializer<Object>) null, (TypeDeserializer) null);
      String[] stringArray0 = new String[0];
      mapDeserializer0.setIgnorableProperties(stringArray0);
      assertTrue(mapDeserializer0.isCachable());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      SimpleType simpleType0 = (SimpleType)TypeBindings.UNBOUND;
      JsonLocationInstantiator jsonLocationInstantiator0 = new JsonLocationInstantiator();
      MapDeserializer mapDeserializer0 = new MapDeserializer(simpleType0, jsonLocationInstantiator0, (KeyDeserializer) null, (JsonDeserializer<Object>) null, (TypeDeserializer) null);
      // Undeclared exception!
      try { 
        mapDeserializer0.resolve((DeserializationContext) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.MapDeserializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      SimpleType simpleType0 = (SimpleType)TypeBindings.UNBOUND;
      ObjectMapper objectMapper0 = new ObjectMapper();
      ObjectMapper.DefaultTyping objectMapper_DefaultTyping0 = ObjectMapper.DefaultTyping.NON_CONCRETE_AND_ARRAYS;
      objectMapper0.enableDefaultTypingAsProperty(objectMapper_DefaultTyping0, "");
      ObjectReader objectReader0 = objectMapper0.reader();
      ObjectReader objectReader1 = objectReader0.withType((Type) simpleType0);
      assertNotSame(objectReader1, objectReader0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      SimpleType simpleType0 = (SimpleType)TypeBindings.UNBOUND;
      JsonLocationInstantiator jsonLocationInstantiator0 = new JsonLocationInstantiator();
      MapDeserializer mapDeserializer0 = new MapDeserializer(simpleType0, jsonLocationInstantiator0, (KeyDeserializer) null, (JsonDeserializer<Object>) null, (TypeDeserializer) null);
      HashSet<String> hashSet0 = new HashSet<String>();
      MapDeserializer mapDeserializer1 = new MapDeserializer(mapDeserializer0, (KeyDeserializer) null, (JsonDeserializer<Object>) null, (TypeDeserializer) null, hashSet0);
      boolean boolean0 = mapDeserializer1.isCachable();
      assertFalse(boolean0);
      assertTrue(mapDeserializer0.isCachable());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      SimpleType simpleType0 = (SimpleType)TypeBindings.UNBOUND;
      JsonLocationInstantiator jsonLocationInstantiator0 = new JsonLocationInstantiator();
      MapDeserializer mapDeserializer0 = new MapDeserializer(simpleType0, jsonLocationInstantiator0, (KeyDeserializer) null, (JsonDeserializer<Object>) null, (TypeDeserializer) null);
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory(deserializerFactoryConfig0);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      SettableBeanProperty[] settableBeanPropertyArray0 = new SettableBeanProperty[0];
      PropertyBasedCreator propertyBasedCreator0 = PropertyBasedCreator.construct(defaultDeserializationContext_Impl0, jsonLocationInstantiator0, settableBeanPropertyArray0);
      mapDeserializer0._propertyBasedCreator = propertyBasedCreator0;
      // Undeclared exception!
      try { 
        mapDeserializer0.deserialize((JsonParser) null, (DeserializationContext) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.MapDeserializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      SimpleType simpleType0 = (SimpleType)TypeBindings.UNBOUND;
      JsonLocationInstantiator jsonLocationInstantiator0 = new JsonLocationInstantiator();
      MapDeserializer mapDeserializer0 = new MapDeserializer(simpleType0, jsonLocationInstantiator0, (KeyDeserializer) null, (JsonDeserializer<Object>) null, (TypeDeserializer) null);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      JsonFactory jsonFactory0 = new JsonFactory();
      JsonParser jsonParser0 = jsonFactory0.createParser((char[]) null, 0, 2);
      HashMap<Object, Object> hashMap0 = new HashMap<Object, Object>();
      // Undeclared exception!
      try { 
        mapDeserializer0.deserialize(jsonParser0, (DeserializationContext) defaultDeserializationContext_Impl0, (Map<Object, Object>) hashMap0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      SimpleType simpleType0 = (SimpleType)TypeBindings.UNBOUND;
      JsonLocationInstantiator jsonLocationInstantiator0 = new JsonLocationInstantiator();
      MapDeserializer mapDeserializer0 = new MapDeserializer(simpleType0, jsonLocationInstantiator0, (KeyDeserializer) null, (JsonDeserializer<Object>) null, (TypeDeserializer) null);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      MapDeserializer mapDeserializer1 = mapDeserializer0.withResolved((KeyDeserializer) null, (TypeDeserializer) null, mapDeserializer0, linkedHashSet0);
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory(true);
      ArrayNode arrayNode0 = new ArrayNode(jsonNodeFactory0);
      JsonParser jsonParser0 = arrayNode0.traverse();
      HashMap<Object, Object> hashMap0 = new HashMap<Object, Object>();
      ObjectMapper objectMapper0 = new ObjectMapper();
      DeserializationContext deserializationContext0 = objectMapper0.getDeserializationContext();
      mapDeserializer1._readAndBind(jsonParser0, deserializationContext0, hashMap0);
      assertFalse(mapDeserializer1.isCachable());
      assertEquals(0, jsonParser0.getCurrentTokenId());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      SimpleType simpleType0 = (SimpleType)TypeBindings.UNBOUND;
      JsonLocationInstantiator jsonLocationInstantiator0 = new JsonLocationInstantiator();
      MapDeserializer mapDeserializer0 = new MapDeserializer(simpleType0, jsonLocationInstantiator0, (KeyDeserializer) null, (JsonDeserializer<Object>) null, (TypeDeserializer) null);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      MapDeserializer mapDeserializer1 = mapDeserializer0.withResolved((KeyDeserializer) null, (TypeDeserializer) null, mapDeserializer0, linkedHashSet0);
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ArrayNode arrayNode0 = jsonNodeFactory0.arrayNode();
      JsonParser jsonParser0 = arrayNode0.traverse();
      HashMap<Object, Object> hashMap0 = new HashMap<Object, Object>();
      ObjectMapper objectMapper0 = new ObjectMapper();
      DeserializationContext deserializationContext0 = objectMapper0.getDeserializationContext();
      mapDeserializer1._readAndBindStringMap(jsonParser0, deserializationContext0, hashMap0);
      assertEquals(0, jsonParser0.getCurrentTokenId());
      assertFalse(mapDeserializer1.isCachable());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      SimpleType simpleType0 = (SimpleType)TypeBindings.UNBOUND;
      JsonLocationInstantiator jsonLocationInstantiator0 = new JsonLocationInstantiator();
      SQLIntegrityConstraintViolationException sQLIntegrityConstraintViolationException0 = new SQLIntegrityConstraintViolationException("|-- unresoZved forward<reference?");
      InvocationTargetException invocationTargetException0 = new InvocationTargetException(sQLIntegrityConstraintViolationException0, "|-- unresoZved forward<reference?");
      MapDeserializer mapDeserializer0 = new MapDeserializer(simpleType0, jsonLocationInstantiator0, (KeyDeserializer) null, (JsonDeserializer<Object>) null, (TypeDeserializer) null);
      // Undeclared exception!
      try { 
        mapDeserializer0.wrapAndThrow(invocationTargetException0, invocationTargetException0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // Can not pass null fieldName
         //
         verifyException("com.fasterxml.jackson.databind.JsonMappingException$Reference", e);
      }
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      SimpleType simpleType0 = (SimpleType)TypeBindings.UNBOUND;
      JsonLocationInstantiator jsonLocationInstantiator0 = new JsonLocationInstantiator();
      MapDeserializer mapDeserializer0 = new MapDeserializer(simpleType0, jsonLocationInstantiator0, (KeyDeserializer) null, (JsonDeserializer<Object>) null, (TypeDeserializer) null);
      InvocationTargetException invocationTargetException0 = new InvocationTargetException((Throwable) null, "1RN!A$!(5 .@@X-G");
      ObjectIdGenerators.IntSequenceGenerator objectIdGenerators_IntSequenceGenerator0 = new ObjectIdGenerators.IntSequenceGenerator();
      // Undeclared exception!
      try { 
        mapDeserializer0.wrapAndThrow(invocationTargetException0, objectIdGenerators_IntSequenceGenerator0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // Can not pass null fieldName
         //
         verifyException("com.fasterxml.jackson.databind.JsonMappingException$Reference", e);
      }
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      SimpleType simpleType0 = (SimpleType)TypeBindings.UNBOUND;
      JsonLocationInstantiator jsonLocationInstantiator0 = new JsonLocationInstantiator();
      SQLIntegrityConstraintViolationException sQLIntegrityConstraintViolationException0 = new SQLIntegrityConstraintViolationException("|-- unresoZved forward<reference?");
      MapDeserializer mapDeserializer0 = new MapDeserializer(simpleType0, jsonLocationInstantiator0, (KeyDeserializer) null, (JsonDeserializer<Object>) null, (TypeDeserializer) null);
      MockError mockError0 = new MockError("|-- unresoZved forward<reference?", sQLIntegrityConstraintViolationException0);
      // Undeclared exception!
      try { 
        mapDeserializer0.wrapAndThrow(mockError0, mockError0);
        fail("Expecting exception: Error");
      
      } catch(Error e) {
      }
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      SimpleType simpleType0 = (SimpleType)TypeBindings.UNBOUND;
      JsonLocationInstantiator jsonLocationInstantiator0 = new JsonLocationInstantiator();
      MapDeserializer mapDeserializer0 = new MapDeserializer(simpleType0, jsonLocationInstantiator0, (KeyDeserializer) null, (JsonDeserializer<Object>) null, (TypeDeserializer) null);
      MockIOException mockIOException0 = new MockIOException(",p[);v");
      try { 
        mapDeserializer0.wrapAndThrow(mockIOException0, simpleType0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
      }
  }
}
