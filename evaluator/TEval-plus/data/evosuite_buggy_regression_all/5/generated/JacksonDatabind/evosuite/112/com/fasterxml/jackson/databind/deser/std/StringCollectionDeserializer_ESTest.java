/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 23:10:47 GMT 2023
 */

package com.fasterxml.jackson.databind.deser.std;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.annotation.PropertyAccessor;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.ObjectCodec;
import com.fasterxml.jackson.core.filter.FilteringParserDelegate;
import com.fasterxml.jackson.core.filter.TokenFilter;
import com.fasterxml.jackson.databind.BeanProperty;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.JsonDeserializer;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.deser.BeanDeserializerFactory;
import com.fasterxml.jackson.databind.deser.DefaultDeserializationContext;
import com.fasterxml.jackson.databind.deser.NullValueProvider;
import com.fasterxml.jackson.databind.deser.ValueInstantiator;
import com.fasterxml.jackson.databind.deser.std.FromStringDeserializer;
import com.fasterxml.jackson.databind.deser.std.StringCollectionDeserializer;
import com.fasterxml.jackson.databind.jsontype.impl.AsExternalTypeDeserializer;
import com.fasterxml.jackson.databind.jsontype.impl.ClassNameIdResolver;
import com.fasterxml.jackson.databind.module.SimpleModule;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import com.fasterxml.jackson.databind.type.MapLikeType;
import com.fasterxml.jackson.databind.type.MapType;
import com.fasterxml.jackson.databind.type.ReferenceType;
import com.fasterxml.jackson.databind.type.TypeFactory;
import java.io.IOException;
import java.time.chrono.ChronoLocalDate;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.Locale;
import java.util.Vector;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class StringCollectionDeserializer_ESTest extends StringCollectionDeserializer_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      StringCollectionDeserializer stringCollectionDeserializer0 = new StringCollectionDeserializer((JavaType) null, (JsonDeserializer<?>) null, (ValueInstantiator) null);
      JsonDeserializer<Object> jsonDeserializer0 = stringCollectionDeserializer0.getContentDeserializer();
      assertNull(jsonDeserializer0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Class<PropertyAccessor> class0 = PropertyAccessor.class;
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(class0);
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      MapLikeType mapLikeType0 = typeFactory0.constructMapLikeType(class0, class0, class0);
      StringCollectionDeserializer stringCollectionDeserializer0 = new StringCollectionDeserializer(mapLikeType0, (JsonDeserializer<?>) null, valueInstantiator_Base0);
      ObjectMapper objectMapper0 = new ObjectMapper();
      DeserializationContext deserializationContext0 = objectMapper0.getDeserializationContext();
      try { 
        stringCollectionDeserializer0.getEmptyValue(deserializationContext0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Cannot create empty instance of [map-like type; class com.fasterxml.jackson.annotation.PropertyAccessor, [simple type, class com.fasterxml.jackson.annotation.PropertyAccessor] -> [simple type, class com.fasterxml.jackson.annotation.PropertyAccessor]], no default Creator
         //
         verifyException("com.fasterxml.jackson.databind.exc.InvalidDefinitionException", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Class<PropertyAccessor> class0 = PropertyAccessor.class;
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(class0);
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<String> class1 = String.class;
      MapLikeType mapLikeType0 = typeFactory0.constructMapLikeType(class0, class0, class1);
      StringCollectionDeserializer stringCollectionDeserializer0 = new StringCollectionDeserializer(mapLikeType0, (JsonDeserializer<?>) null, valueInstantiator_Base0);
      ObjectMapper objectMapper0 = new ObjectMapper();
      DeserializationContext deserializationContext0 = objectMapper0.getDeserializationContext();
      JsonFactory jsonFactory0 = new JsonFactory();
      byte[] byteArray0 = new byte[8];
      JsonParser jsonParser0 = jsonFactory0.createParser(byteArray0, (int) (byte)0, (int) (byte)1);
      ClassNameIdResolver classNameIdResolver0 = new ClassNameIdResolver(mapLikeType0, typeFactory0);
      AsExternalTypeDeserializer asExternalTypeDeserializer0 = new AsExternalTypeDeserializer(mapLikeType0, classNameIdResolver0, "", true, mapLikeType0);
      try { 
        stringCollectionDeserializer0.deserializeWithType(jsonParser0, deserializationContext0, asExternalTypeDeserializer0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Could not resolve type id 'com.fasterxml.jackson.annotation.PropertyAccessor' as a subtype of [map-like type; class com.fasterxml.jackson.annotation.PropertyAccessor, [simple type, class com.fasterxml.jackson.annotation.PropertyAccessor] -> [simple type, class java.lang.String]]: problem: (java.lang.NullPointerException) null
         //
         verifyException("com.fasterxml.jackson.databind.exc.InvalidTypeIdException", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      BeanProperty.Bogus beanProperty_Bogus0 = new BeanProperty.Bogus();
      JavaType javaType0 = beanProperty_Bogus0.getType();
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Boolean boolean0 = Boolean.FALSE;
      JsonDeserializer<SimpleModule> jsonDeserializer0 = (JsonDeserializer<SimpleModule>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      StringCollectionDeserializer stringCollectionDeserializer0 = new StringCollectionDeserializer(javaType0, (ValueInstantiator) null, jsonDeserializer0, jsonDeserializer0, jsonDeserializer0, boolean0);
      JsonDeserializer<?> jsonDeserializer1 = stringCollectionDeserializer0.createContextual(defaultDeserializationContext_Impl0, beanProperty_Bogus0);
      assertFalse(jsonDeserializer1.isCachable());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Class<ChronoLocalDate> class0 = ChronoLocalDate.class;
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(class0);
      StringCollectionDeserializer stringCollectionDeserializer0 = new StringCollectionDeserializer((JavaType) null, valueInstantiator_Base0, (JsonDeserializer<?>) null, (JsonDeserializer<?>) null, (NullValueProvider) null, (Boolean) null);
      StringCollectionDeserializer stringCollectionDeserializer1 = stringCollectionDeserializer0.withResolved((JsonDeserializer<?>) null, (JsonDeserializer<?>) null, (NullValueProvider) null, (Boolean) null);
      assertSame(stringCollectionDeserializer1, stringCollectionDeserializer0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<HashMap> class0 = HashMap.class;
      MapType mapType0 = typeFactory0.constructRawMapType(class0);
      ReferenceType referenceType0 = ReferenceType.upgradeFrom(mapType0, mapType0);
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(referenceType0);
      Boolean boolean0 = Boolean.TRUE;
      StringCollectionDeserializer stringCollectionDeserializer0 = new StringCollectionDeserializer(referenceType0, valueInstantiator_Base0, (JsonDeserializer<?>) null, (JsonDeserializer<?>) null, (NullValueProvider) null, boolean0);
      StringCollectionDeserializer stringCollectionDeserializer1 = new StringCollectionDeserializer(mapType0, valueInstantiator_Base0, stringCollectionDeserializer0, stringCollectionDeserializer0, stringCollectionDeserializer0, boolean0);
      StringCollectionDeserializer stringCollectionDeserializer2 = stringCollectionDeserializer1.withResolved(stringCollectionDeserializer0, stringCollectionDeserializer0, (NullValueProvider) null, boolean0);
      assertNotSame(stringCollectionDeserializer2, stringCollectionDeserializer1);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<HashMap> class0 = HashMap.class;
      MapType mapType0 = typeFactory0.constructRawMapType(class0);
      ReferenceType referenceType0 = ReferenceType.upgradeFrom(mapType0, mapType0);
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(referenceType0);
      Boolean boolean0 = Boolean.FALSE;
      StringCollectionDeserializer stringCollectionDeserializer0 = new StringCollectionDeserializer(referenceType0, valueInstantiator_Base0, (JsonDeserializer<?>) null, (JsonDeserializer<?>) null, (NullValueProvider) null, boolean0);
      StringCollectionDeserializer stringCollectionDeserializer1 = stringCollectionDeserializer0.withResolved(stringCollectionDeserializer0, stringCollectionDeserializer0, (NullValueProvider) null, boolean0);
      assertFalse(stringCollectionDeserializer1.equals((Object)stringCollectionDeserializer0));
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      BeanProperty.Bogus beanProperty_Bogus0 = new BeanProperty.Bogus();
      JavaType javaType0 = beanProperty_Bogus0.getType();
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      JsonDeserializer<Locale.LanguageRange> jsonDeserializer0 = (JsonDeserializer<Locale.LanguageRange>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      StringCollectionDeserializer stringCollectionDeserializer0 = new StringCollectionDeserializer(javaType0, (ValueInstantiator) null, jsonDeserializer0, jsonDeserializer0, jsonDeserializer0, (Boolean) null);
      JsonDeserializer<?> jsonDeserializer1 = stringCollectionDeserializer0.createContextual(defaultDeserializationContext_Impl0, beanProperty_Bogus0);
      assertFalse(jsonDeserializer1.isCachable());
      assertNotSame(stringCollectionDeserializer0, jsonDeserializer1);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Class<PropertyAccessor> class0 = PropertyAccessor.class;
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(class0);
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      MapLikeType mapLikeType0 = typeFactory0.constructMapLikeType(class0, class0, class0);
      Boolean boolean0 = Boolean.valueOf(true);
      Class<String> class1 = String.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class1, (-1601));
      StringCollectionDeserializer stringCollectionDeserializer0 = new StringCollectionDeserializer(mapLikeType0, valueInstantiator_Base0, fromStringDeserializer_Std0, fromStringDeserializer_Std0, fromStringDeserializer_Std0, boolean0);
      boolean boolean1 = stringCollectionDeserializer0.isCachable();
      assertFalse(boolean1);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Class<PropertyAccessor> class0 = PropertyAccessor.class;
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(class0);
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      MapLikeType mapLikeType0 = typeFactory0.constructMapLikeType(class0, class0, class0);
      StringCollectionDeserializer stringCollectionDeserializer0 = new StringCollectionDeserializer(mapLikeType0, (JsonDeserializer<?>) null, valueInstantiator_Base0);
      boolean boolean0 = stringCollectionDeserializer0.isCachable();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Class<PropertyAccessor> class0 = PropertyAccessor.class;
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(class0);
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<String> class1 = String.class;
      MapLikeType mapLikeType0 = typeFactory0.constructMapLikeType(class0, class0, class1);
      StringCollectionDeserializer stringCollectionDeserializer0 = new StringCollectionDeserializer(mapLikeType0, (JsonDeserializer<?>) null, valueInstantiator_Base0);
      ReferenceType referenceType0 = ReferenceType.upgradeFrom(mapLikeType0, mapLikeType0);
      StringCollectionDeserializer stringCollectionDeserializer1 = new StringCollectionDeserializer(referenceType0, valueInstantiator_Base0, stringCollectionDeserializer0, (JsonDeserializer<?>) null, stringCollectionDeserializer0, (Boolean) null);
      boolean boolean0 = stringCollectionDeserializer1.isCachable();
      assertTrue(stringCollectionDeserializer0.isCachable());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Class<PropertyAccessor> class0 = PropertyAccessor.class;
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(class0);
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      MapLikeType mapLikeType0 = typeFactory0.constructMapLikeType(class0, class0, class0);
      StringCollectionDeserializer stringCollectionDeserializer0 = new StringCollectionDeserializer(mapLikeType0, (JsonDeserializer<?>) null, valueInstantiator_Base0);
      ObjectMapper objectMapper0 = new ObjectMapper();
      DeserializationContext deserializationContext0 = objectMapper0.getDeserializationContext();
      // Undeclared exception!
      try { 
        stringCollectionDeserializer0.createContextual(deserializationContext0, (BeanProperty) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      ArrayNode arrayNode0 = objectMapper0.createArrayNode();
      JsonParser jsonParser0 = arrayNode0.traverse((ObjectCodec) objectMapper0);
      TokenFilter tokenFilter0 = TokenFilter.INCLUDE_ALL;
      FilteringParserDelegate filteringParserDelegate0 = new FilteringParserDelegate(jsonParser0, tokenFilter0, true, true);
      filteringParserDelegate0.nextBooleanValue();
      StringCollectionDeserializer stringCollectionDeserializer0 = new StringCollectionDeserializer((JavaType) null, (ValueInstantiator) null, (JsonDeserializer<?>) null, (JsonDeserializer<?>) null, (NullValueProvider) null, (Boolean) null);
      DeserializationContext deserializationContext0 = objectMapper0.getDeserializationContext();
      StringCollectionDeserializer stringCollectionDeserializer1 = new StringCollectionDeserializer((JavaType) null, stringCollectionDeserializer0, (ValueInstantiator) null);
      HashSet<String> hashSet0 = new HashSet<String>();
      Collection<String> collection0 = stringCollectionDeserializer0.deserialize(jsonParser0, deserializationContext0, (Collection<String>) hashSet0);
      // Undeclared exception!
      try { 
        stringCollectionDeserializer1.deserialize((JsonParser) filteringParserDelegate0, deserializationContext0, collection0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.StringCollectionDeserializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      BeanProperty.Bogus beanProperty_Bogus0 = new BeanProperty.Bogus();
      JavaType javaType0 = beanProperty_Bogus0.getType();
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Boolean boolean0 = Boolean.FALSE;
      JsonDeserializer<SimpleModule> jsonDeserializer0 = (JsonDeserializer<SimpleModule>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      doReturn((Object) null).when(jsonDeserializer0).deserialize(any(com.fasterxml.jackson.core.JsonParser.class) , any(com.fasterxml.jackson.databind.DeserializationContext.class));
      StringCollectionDeserializer stringCollectionDeserializer0 = new StringCollectionDeserializer(javaType0, (ValueInstantiator) null, jsonDeserializer0, jsonDeserializer0, jsonDeserializer0, boolean0);
      // Undeclared exception!
      try { 
        stringCollectionDeserializer0.deserialize((JsonParser) null, (DeserializationContext) defaultDeserializationContext_Impl0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.StringCollectionDeserializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Class<SimpleModule> class0 = SimpleModule.class;
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(class0);
      ObjectMapper objectMapper0 = new ObjectMapper();
      ArrayNode arrayNode0 = objectMapper0.createArrayNode();
      JsonParser jsonParser0 = arrayNode0.traverse((ObjectCodec) objectMapper0);
      Boolean boolean0 = Boolean.valueOf(false);
      StringCollectionDeserializer stringCollectionDeserializer0 = new StringCollectionDeserializer((JavaType) null, valueInstantiator_Base0, (JsonDeserializer<?>) null, (JsonDeserializer<?>) null, (NullValueProvider) null, boolean0);
      DeserializationContext deserializationContext0 = objectMapper0.getDeserializationContext();
      Vector<String> vector0 = new Vector<String>();
      // Undeclared exception!
      try { 
        stringCollectionDeserializer0.deserialize(jsonParser0, deserializationContext0, (Collection<String>) vector0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.StringCollectionDeserializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      ArrayNode arrayNode0 = objectMapper0.createArrayNode();
      arrayNode0.insert((-1822), (Boolean) null);
      JsonParser jsonParser0 = arrayNode0.traverse((ObjectCodec) objectMapper0);
      TokenFilter tokenFilter0 = TokenFilter.INCLUDE_ALL;
      FilteringParserDelegate filteringParserDelegate0 = new FilteringParserDelegate(jsonParser0, tokenFilter0, true, true);
      filteringParserDelegate0.nextBooleanValue();
      StringCollectionDeserializer stringCollectionDeserializer0 = new StringCollectionDeserializer((JavaType) null, (ValueInstantiator) null, (JsonDeserializer<?>) null, (JsonDeserializer<?>) null, (NullValueProvider) null, (Boolean) null);
      DeserializationContext deserializationContext0 = objectMapper0.getDeserializationContext();
      HashSet<String> hashSet0 = new HashSet<String>();
      try { 
        stringCollectionDeserializer0.deserialize((JsonParser) filteringParserDelegate0, deserializationContext0, (Collection<String>) hashSet0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // (was java.lang.NullPointerException) (through reference chain: java.util.HashSet[0])
         //
         verifyException("com.fasterxml.jackson.databind.JsonMappingException", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      ArrayNode arrayNode0 = objectMapper0.createArrayNode();
      JsonParser jsonParser0 = arrayNode0.traverse((ObjectCodec) objectMapper0);
      TokenFilter tokenFilter0 = TokenFilter.INCLUDE_ALL;
      FilteringParserDelegate filteringParserDelegate0 = new FilteringParserDelegate(jsonParser0, tokenFilter0, true, true);
      filteringParserDelegate0.nextBooleanValue();
      StringCollectionDeserializer stringCollectionDeserializer0 = new StringCollectionDeserializer((JavaType) null, (ValueInstantiator) null, (JsonDeserializer<?>) null, (JsonDeserializer<?>) null, (NullValueProvider) null, (Boolean) null);
      DeserializationContext deserializationContext0 = objectMapper0.getDeserializationContext();
      stringCollectionDeserializer0.deserialize(jsonParser0, deserializationContext0, (Collection<String>) null);
      // Undeclared exception!
      try { 
        stringCollectionDeserializer0.deserialize((JsonParser) filteringParserDelegate0, deserializationContext0, (Collection<String>) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.StringCollectionDeserializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      ArrayNode arrayNode0 = objectMapper0.createArrayNode();
      arrayNode0.insert((-1822), (Boolean) null);
      JsonParser jsonParser0 = arrayNode0.traverse((ObjectCodec) objectMapper0);
      TokenFilter tokenFilter0 = TokenFilter.INCLUDE_ALL;
      FilteringParserDelegate filteringParserDelegate0 = new FilteringParserDelegate(jsonParser0, tokenFilter0, true, true);
      filteringParserDelegate0.nextBooleanValue();
      StringCollectionDeserializer stringCollectionDeserializer0 = new StringCollectionDeserializer((JavaType) null, (ValueInstantiator) null, (JsonDeserializer<?>) null, (JsonDeserializer<?>) null, (NullValueProvider) null, (Boolean) null);
      DeserializationContext deserializationContext0 = objectMapper0.getDeserializationContext();
      StringCollectionDeserializer stringCollectionDeserializer1 = new StringCollectionDeserializer((JavaType) null, stringCollectionDeserializer0, (ValueInstantiator) null);
      HashSet<String> hashSet0 = new HashSet<String>();
      stringCollectionDeserializer1.deserialize((JsonParser) filteringParserDelegate0, deserializationContext0, (Collection<String>) hashSet0);
      assertEquals(1, hashSet0.size());
      assertFalse(stringCollectionDeserializer1.isCachable());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Class<PropertyAccessor> class0 = PropertyAccessor.class;
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(class0);
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      MapLikeType mapLikeType0 = typeFactory0.constructMapLikeType(class0, class0, class0);
      ObjectMapper objectMapper0 = new ObjectMapper();
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ArrayNode arrayNode0 = jsonNodeFactory0.arrayNode();
      JsonParser jsonParser0 = arrayNode0.traverse((ObjectCodec) objectMapper0);
      HashSet<String> hashSet0 = new HashSet<String>();
      DeserializationContext deserializationContext0 = objectMapper0.getDeserializationContext();
      Boolean boolean0 = Boolean.valueOf(true);
      Class<String> class1 = String.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class1, (-1601));
      StringCollectionDeserializer stringCollectionDeserializer0 = new StringCollectionDeserializer(mapLikeType0, valueInstantiator_Base0, fromStringDeserializer_Std0, fromStringDeserializer_Std0, fromStringDeserializer_Std0, boolean0);
      // Undeclared exception!
      try { 
        stringCollectionDeserializer0.deserialize(jsonParser0, deserializationContext0, (Collection<String>) hashSet0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ArrayNode arrayNode0 = jsonNodeFactory0.arrayNode();
      JsonParser jsonParser0 = arrayNode0.traverse((ObjectCodec) objectMapper0);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      DeserializationContext deserializationContext0 = objectMapper0.getDeserializationContext();
      StringCollectionDeserializer stringCollectionDeserializer0 = new StringCollectionDeserializer((JavaType) null, (JsonDeserializer<?>) null, (ValueInstantiator) null);
      // Undeclared exception!
      try { 
        stringCollectionDeserializer0.deserialize(jsonParser0, deserializationContext0, (Collection<String>) linkedHashSet0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.StringCollectionDeserializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Class<SimpleModule> class0 = SimpleModule.class;
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(class0);
      ObjectMapper objectMapper0 = new ObjectMapper();
      ArrayNode arrayNode0 = objectMapper0.createArrayNode();
      JsonParser jsonParser0 = arrayNode0.traverse((ObjectCodec) objectMapper0);
      Boolean boolean0 = Boolean.valueOf(true);
      StringCollectionDeserializer stringCollectionDeserializer0 = new StringCollectionDeserializer((JavaType) null, valueInstantiator_Base0, (JsonDeserializer<?>) null, (JsonDeserializer<?>) null, (NullValueProvider) null, boolean0);
      DeserializationContext deserializationContext0 = objectMapper0.getDeserializationContext();
      Vector<String> vector0 = new Vector<String>();
      // Undeclared exception!
      try { 
        stringCollectionDeserializer0.deserialize(jsonParser0, deserializationContext0, (Collection<String>) vector0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }
}