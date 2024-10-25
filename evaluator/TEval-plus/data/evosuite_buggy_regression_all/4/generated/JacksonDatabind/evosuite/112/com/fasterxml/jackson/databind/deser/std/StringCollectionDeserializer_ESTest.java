/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 21:49:25 GMT 2023
 */

package com.fasterxml.jackson.databind.deser.std;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.annotation.ObjectIdGenerators;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.BeanProperty;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.JsonDeserializer;
import com.fasterxml.jackson.databind.deser.BeanDeserializerFactory;
import com.fasterxml.jackson.databind.deser.CreatorProperty;
import com.fasterxml.jackson.databind.deser.DefaultDeserializationContext;
import com.fasterxml.jackson.databind.deser.NullValueProvider;
import com.fasterxml.jackson.databind.deser.ValueInstantiator;
import com.fasterxml.jackson.databind.deser.std.StringCollectionDeserializer;
import com.fasterxml.jackson.databind.ext.CoreXMLDeserializers;
import com.fasterxml.jackson.databind.jsontype.TypeIdResolver;
import com.fasterxml.jackson.databind.jsontype.impl.AsPropertyTypeDeserializer;
import com.fasterxml.jackson.databind.type.MapLikeType;
import com.fasterxml.jackson.databind.type.SimpleType;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.time.DayOfWeek;
import java.util.Collection;
import java.util.Stack;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class StringCollectionDeserializer_ESTest extends StringCollectionDeserializer_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Class<ByteArrayInputStream> class0 = ByteArrayInputStream.class;
      CoreXMLDeserializers.Std coreXMLDeserializers_Std0 = new CoreXMLDeserializers.Std(class0, 1);
      StringCollectionDeserializer stringCollectionDeserializer0 = new StringCollectionDeserializer((JavaType) null, (ValueInstantiator) null, coreXMLDeserializers_Std0, coreXMLDeserializers_Std0, coreXMLDeserializers_Std0, (Boolean) null);
      // Undeclared exception!
      try { 
        stringCollectionDeserializer0.findBackReference("JSON");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Cannot handle managed/back reference 'JSON': type: value deserializer of type com.fasterxml.jackson.databind.ext.CoreXMLDeserializers$Std does not support them
         //
         verifyException("com.fasterxml.jackson.databind.JsonDeserializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      StringCollectionDeserializer stringCollectionDeserializer0 = new StringCollectionDeserializer((JavaType) null, (JsonDeserializer<?>) null, (ValueInstantiator) null);
      try { 
        stringCollectionDeserializer0.getEmptyValue((DeserializationContext) defaultDeserializationContext_Impl0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Cannot create empty instance of null, no default Creator
         //
         verifyException("com.fasterxml.jackson.databind.exc.InvalidDefinitionException", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      BeanProperty.Bogus beanProperty_Bogus0 = new BeanProperty.Bogus();
      JavaType javaType0 = beanProperty_Bogus0.getType();
      JsonDeserializer<Integer> jsonDeserializer0 = (JsonDeserializer<Integer>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      Boolean boolean0 = new Boolean("");
      StringCollectionDeserializer stringCollectionDeserializer0 = new StringCollectionDeserializer(javaType0, (ValueInstantiator) null, jsonDeserializer0, jsonDeserializer0, jsonDeserializer0, boolean0);
      JsonFactory jsonFactory0 = new JsonFactory();
      char[] charArray0 = new char[2];
      JsonParser jsonParser0 = jsonFactory0.createParser(charArray0, 0, 1);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      AsPropertyTypeDeserializer asPropertyTypeDeserializer0 = new AsPropertyTypeDeserializer(javaType0, (TypeIdResolver) null, "JSON", false, javaType0);
      // Undeclared exception!
      try { 
        stringCollectionDeserializer0.deserializeWithType(jsonParser0, defaultDeserializationContext_Impl0, asPropertyTypeDeserializer0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.jsontype.impl.AsArrayTypeDeserializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Boolean boolean0 = Boolean.TRUE;
      JsonDeserializer<ObjectIdGenerators.StringIdGenerator> jsonDeserializer0 = (JsonDeserializer<ObjectIdGenerators.StringIdGenerator>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      StringCollectionDeserializer stringCollectionDeserializer0 = new StringCollectionDeserializer((JavaType) null, (ValueInstantiator) null, jsonDeserializer0, jsonDeserializer0, jsonDeserializer0, boolean0);
      StringCollectionDeserializer stringCollectionDeserializer1 = new StringCollectionDeserializer((JavaType) null, stringCollectionDeserializer0, (ValueInstantiator) null);
      StringCollectionDeserializer stringCollectionDeserializer2 = stringCollectionDeserializer1.withResolved(stringCollectionDeserializer0, stringCollectionDeserializer0, stringCollectionDeserializer0, (Boolean) null);
      assertNotSame(stringCollectionDeserializer2, stringCollectionDeserializer1);
      assertFalse(stringCollectionDeserializer1.isCachable());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Class<String> class0 = String.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(simpleType0);
      StringCollectionDeserializer stringCollectionDeserializer0 = new StringCollectionDeserializer(simpleType0, valueInstantiator_Base0, (JsonDeserializer<?>) null, (JsonDeserializer<?>) null, (NullValueProvider) null, (Boolean) null);
      StringCollectionDeserializer stringCollectionDeserializer1 = stringCollectionDeserializer0.withResolved((JsonDeserializer<?>) null, stringCollectionDeserializer0, stringCollectionDeserializer0, (Boolean) null);
      boolean boolean0 = stringCollectionDeserializer1.isCachable();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Class<ByteArrayInputStream> class0 = ByteArrayInputStream.class;
      CoreXMLDeserializers.Std coreXMLDeserializers_Std0 = new CoreXMLDeserializers.Std(class0, 1);
      CoreXMLDeserializers.Std coreXMLDeserializers_Std1 = new CoreXMLDeserializers.Std(class0, 1);
      StringCollectionDeserializer stringCollectionDeserializer0 = new StringCollectionDeserializer((JavaType) null, (ValueInstantiator) null, coreXMLDeserializers_Std0, coreXMLDeserializers_Std1, coreXMLDeserializers_Std0, (Boolean) null);
      StringCollectionDeserializer stringCollectionDeserializer1 = stringCollectionDeserializer0.withResolved(coreXMLDeserializers_Std0, coreXMLDeserializers_Std0, coreXMLDeserializers_Std0, (Boolean) null);
      assertNotSame(stringCollectionDeserializer1, stringCollectionDeserializer0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      StringCollectionDeserializer stringCollectionDeserializer0 = new StringCollectionDeserializer((JavaType) null, (JsonDeserializer<?>) null, (ValueInstantiator) null);
      StringCollectionDeserializer stringCollectionDeserializer1 = stringCollectionDeserializer0.withResolved((JsonDeserializer<?>) null, (JsonDeserializer<?>) null, (NullValueProvider) null, (Boolean) null);
      assertSame(stringCollectionDeserializer1, stringCollectionDeserializer0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Class<String> class0 = String.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      MapLikeType mapLikeType0 = MapLikeType.upgradeFrom(simpleType0, simpleType0, simpleType0);
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(simpleType0);
      Boolean boolean0 = Boolean.FALSE;
      StringCollectionDeserializer stringCollectionDeserializer0 = new StringCollectionDeserializer(mapLikeType0, valueInstantiator_Base0, (JsonDeserializer<?>) null, (JsonDeserializer<?>) null, (NullValueProvider) null, boolean0);
      boolean boolean1 = stringCollectionDeserializer0.isCachable();
      assertTrue(boolean1);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Class<String> class0 = String.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      MapLikeType mapLikeType0 = MapLikeType.upgradeFrom(simpleType0, simpleType0, simpleType0);
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(simpleType0);
      Boolean boolean0 = Boolean.FALSE;
      StringCollectionDeserializer stringCollectionDeserializer0 = new StringCollectionDeserializer(mapLikeType0, valueInstantiator_Base0, (JsonDeserializer<?>) null, (JsonDeserializer<?>) null, (NullValueProvider) null, boolean0);
      JsonDeserializer<ByteArrayInputStream> jsonDeserializer0 = (JsonDeserializer<ByteArrayInputStream>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      StringCollectionDeserializer stringCollectionDeserializer1 = stringCollectionDeserializer0.withResolved(jsonDeserializer0, (JsonDeserializer<?>) null, (NullValueProvider) null, (Boolean) null);
      boolean boolean1 = stringCollectionDeserializer1.isCachable();
      assertFalse(boolean1);
      assertTrue(stringCollectionDeserializer0.isCachable());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Boolean boolean0 = Boolean.TRUE;
      JsonDeserializer<ObjectIdGenerators.StringIdGenerator> jsonDeserializer0 = (JsonDeserializer<ObjectIdGenerators.StringIdGenerator>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      StringCollectionDeserializer stringCollectionDeserializer0 = new StringCollectionDeserializer((JavaType) null, (ValueInstantiator) null, jsonDeserializer0, jsonDeserializer0, jsonDeserializer0, boolean0);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      BeanProperty.Bogus beanProperty_Bogus0 = new BeanProperty.Bogus();
      // Undeclared exception!
      try { 
        stringCollectionDeserializer0.createContextual(defaultDeserializationContext_Impl0, beanProperty_Bogus0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.StringCollectionDeserializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Class<InputStream> class0 = InputStream.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(class0);
      Boolean boolean0 = Boolean.FALSE;
      StringCollectionDeserializer stringCollectionDeserializer0 = new StringCollectionDeserializer(simpleType0, valueInstantiator_Base0, (JsonDeserializer<?>) null, (JsonDeserializer<?>) null, (NullValueProvider) null, boolean0);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      BeanProperty.Bogus beanProperty_Bogus0 = new BeanProperty.Bogus();
      // Undeclared exception!
      try { 
        stringCollectionDeserializer0.createContextual(defaultDeserializationContext_Impl0, beanProperty_Bogus0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Class<DayOfWeek> class0 = DayOfWeek.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(class0);
      Boolean boolean0 = Boolean.TRUE;
      JsonDeserializer<CreatorProperty> jsonDeserializer0 = (JsonDeserializer<CreatorProperty>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      StringCollectionDeserializer stringCollectionDeserializer0 = new StringCollectionDeserializer(simpleType0, valueInstantiator_Base0, jsonDeserializer0, jsonDeserializer0, jsonDeserializer0, boolean0);
      StringCollectionDeserializer stringCollectionDeserializer1 = new StringCollectionDeserializer(simpleType0, valueInstantiator_Base0, stringCollectionDeserializer0, stringCollectionDeserializer0, stringCollectionDeserializer0, boolean0);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      BeanProperty.Bogus beanProperty_Bogus0 = new BeanProperty.Bogus();
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      JsonDeserializer<?> jsonDeserializer1 = stringCollectionDeserializer1.createContextual(defaultDeserializationContext_Impl0, beanProperty_Bogus0);
      assertTrue(jsonDeserializer1.isCachable());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Class<InputStream> class0 = InputStream.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(class0);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      StringCollectionDeserializer stringCollectionDeserializer0 = new StringCollectionDeserializer(simpleType0, (JsonDeserializer<?>) null, valueInstantiator_Base0);
      // Undeclared exception!
      try { 
        stringCollectionDeserializer0.deserialize((JsonParser) null, (DeserializationContext) defaultDeserializationContext_Impl0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Class<DayOfWeek> class0 = DayOfWeek.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(class0);
      Boolean boolean0 = Boolean.TRUE;
      JsonDeserializer<CreatorProperty> jsonDeserializer0 = (JsonDeserializer<CreatorProperty>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      doReturn((Object) null).when(jsonDeserializer0).deserialize(any(com.fasterxml.jackson.core.JsonParser.class) , any(com.fasterxml.jackson.databind.DeserializationContext.class));
      StringCollectionDeserializer stringCollectionDeserializer0 = new StringCollectionDeserializer(simpleType0, valueInstantiator_Base0, jsonDeserializer0, jsonDeserializer0, jsonDeserializer0, boolean0);
      JsonFactory jsonFactory0 = new JsonFactory();
      JsonParser jsonParser0 = jsonFactory0.createParser("JSON");
      StringCollectionDeserializer stringCollectionDeserializer1 = new StringCollectionDeserializer(simpleType0, valueInstantiator_Base0, stringCollectionDeserializer0, stringCollectionDeserializer0, stringCollectionDeserializer0, boolean0);
      // Undeclared exception!
      try { 
        stringCollectionDeserializer1.deserialize(jsonParser0, (DeserializationContext) null, (Collection<String>) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.ValueInstantiator", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Class<DayOfWeek> class0 = DayOfWeek.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      StringCollectionDeserializer stringCollectionDeserializer0 = new StringCollectionDeserializer(simpleType0, (JsonDeserializer<?>) null, (ValueInstantiator) null);
      JsonFactory jsonFactory0 = new JsonFactory();
      JsonParser jsonParser0 = jsonFactory0.createParser("m");
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Stack<String> stack0 = new Stack<String>();
      // Undeclared exception!
      try { 
        stringCollectionDeserializer0.deserialize(jsonParser0, (DeserializationContext) defaultDeserializationContext_Impl0, (Collection<String>) stack0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Class<InputStream> class0 = InputStream.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(class0);
      Boolean boolean0 = Boolean.FALSE;
      StringCollectionDeserializer stringCollectionDeserializer0 = new StringCollectionDeserializer(simpleType0, valueInstantiator_Base0, (JsonDeserializer<?>) null, (JsonDeserializer<?>) null, (NullValueProvider) null, boolean0);
      JsonFactory jsonFactory0 = new JsonFactory();
      JsonParser jsonParser0 = jsonFactory0.createNonBlockingByteArrayParser();
      Stack<String> stack0 = new Stack<String>();
      // Undeclared exception!
      try { 
        stringCollectionDeserializer0.deserialize(jsonParser0, (DeserializationContext) null, (Collection<String>) stack0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.StringCollectionDeserializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Boolean boolean0 = Boolean.TRUE;
      StringCollectionDeserializer stringCollectionDeserializer0 = new StringCollectionDeserializer((JavaType) null, (ValueInstantiator) null, (JsonDeserializer<?>) null, (JsonDeserializer<?>) null, (NullValueProvider) null, boolean0);
      JsonFactory jsonFactory0 = new JsonFactory();
      JsonParser jsonParser0 = jsonFactory0.createNonBlockingByteArrayParser();
      // Undeclared exception!
      try { 
        stringCollectionDeserializer0.deserialize(jsonParser0, (DeserializationContext) null, (Collection<String>) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.StdDeserializer", e);
      }
  }
}
