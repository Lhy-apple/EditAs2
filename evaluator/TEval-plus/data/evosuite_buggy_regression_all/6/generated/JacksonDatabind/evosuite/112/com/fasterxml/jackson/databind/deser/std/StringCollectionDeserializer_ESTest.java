/*
 * This file was automatically generated by EvoSuite
 * Wed Sep 27 00:31:14 GMT 2023
 */

package com.fasterxml.jackson.databind.deser.std;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.util.JsonParserDelegate;
import com.fasterxml.jackson.databind.BeanProperty;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.JsonDeserializer;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
import com.fasterxml.jackson.databind.deser.BeanDeserializerFactory;
import com.fasterxml.jackson.databind.deser.DefaultDeserializationContext;
import com.fasterxml.jackson.databind.deser.NullValueProvider;
import com.fasterxml.jackson.databind.deser.ValueInstantiator;
import com.fasterxml.jackson.databind.deser.std.StringCollectionDeserializer;
import com.fasterxml.jackson.databind.introspect.BasicBeanDescription;
import com.fasterxml.jackson.databind.jsontype.impl.AsPropertyTypeDeserializer;
import com.fasterxml.jackson.databind.jsontype.impl.ClassNameIdResolver;
import com.fasterxml.jackson.databind.type.SimpleType;
import com.fasterxml.jackson.databind.type.TypeFactory;
import java.math.RoundingMode;
import java.util.ArrayList;
import java.util.Collection;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class StringCollectionDeserializer_ESTest extends StringCollectionDeserializer_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Class<String> class0 = String.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(simpleType0);
      StringCollectionDeserializer stringCollectionDeserializer0 = new StringCollectionDeserializer(simpleType0, (JsonDeserializer<?>) null, valueInstantiator_Base0);
      JsonDeserializer<Object> jsonDeserializer0 = stringCollectionDeserializer0.getContentDeserializer();
      assertNull(jsonDeserializer0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Class<String> class0 = String.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(simpleType0);
      StringCollectionDeserializer stringCollectionDeserializer0 = new StringCollectionDeserializer(simpleType0, (JsonDeserializer<?>) null, valueInstantiator_Base0);
      // Undeclared exception!
      try { 
        stringCollectionDeserializer0.getEmptyValue((DeserializationContext) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.ContainerDeserializerBase", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Class<BasicBeanDescription> class0 = BasicBeanDescription.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(simpleType0);
      JsonDeserializer<Integer> jsonDeserializer0 = (JsonDeserializer<Integer>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      Boolean boolean0 = new Boolean((String) null);
      StringCollectionDeserializer stringCollectionDeserializer0 = new StringCollectionDeserializer(simpleType0, valueInstantiator_Base0, jsonDeserializer0, jsonDeserializer0, jsonDeserializer0, boolean0);
      JsonParserDelegate jsonParserDelegate0 = new JsonParserDelegate((JsonParser) null);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      ClassNameIdResolver classNameIdResolver0 = new ClassNameIdResolver(simpleType0, (TypeFactory) null);
      AsPropertyTypeDeserializer asPropertyTypeDeserializer0 = new AsPropertyTypeDeserializer(simpleType0, classNameIdResolver0, "oY2z*&?\"{.\"?[Nv", false, simpleType0);
      // Undeclared exception!
      try { 
        stringCollectionDeserializer0.deserializeWithType(jsonParserDelegate0, defaultDeserializationContext_Impl0, asPropertyTypeDeserializer0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.core.util.JsonParserDelegate", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Class<String> class0 = String.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(simpleType0);
      StringCollectionDeserializer stringCollectionDeserializer0 = new StringCollectionDeserializer(simpleType0, (JsonDeserializer<?>) null, valueInstantiator_Base0);
      Boolean boolean0 = Boolean.valueOf("DYW8<=<FV]%y/");
      StringCollectionDeserializer stringCollectionDeserializer1 = stringCollectionDeserializer0.withResolved((JsonDeserializer<?>) null, (JsonDeserializer<?>) null, (NullValueProvider) null, boolean0);
      StringCollectionDeserializer stringCollectionDeserializer2 = stringCollectionDeserializer1.withResolved((JsonDeserializer<?>) null, stringCollectionDeserializer0, (NullValueProvider) null, boolean0);
      assertNotSame(stringCollectionDeserializer1, stringCollectionDeserializer0);
      assertFalse(stringCollectionDeserializer2.isCachable());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Class<BasicBeanDescription> class0 = BasicBeanDescription.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      ObjectMapper objectMapper0 = new ObjectMapper();
      ObjectReader objectReader0 = objectMapper0.readerFor((JavaType) simpleType0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Class<String> class0 = String.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(simpleType0);
      StringCollectionDeserializer stringCollectionDeserializer0 = new StringCollectionDeserializer(simpleType0, (JsonDeserializer<?>) null, valueInstantiator_Base0);
      Boolean boolean0 = Boolean.TRUE;
      StringCollectionDeserializer stringCollectionDeserializer1 = stringCollectionDeserializer0.withResolved((JsonDeserializer<?>) null, (JsonDeserializer<?>) null, (NullValueProvider) null, boolean0);
      StringCollectionDeserializer stringCollectionDeserializer2 = new StringCollectionDeserializer(simpleType0, valueInstantiator_Base0, stringCollectionDeserializer1, stringCollectionDeserializer0, (NullValueProvider) null, boolean0);
      StringCollectionDeserializer stringCollectionDeserializer3 = stringCollectionDeserializer2.withResolved(stringCollectionDeserializer0, stringCollectionDeserializer0, (NullValueProvider) null, boolean0);
      assertNotSame(stringCollectionDeserializer3, stringCollectionDeserializer2);
      assertNotSame(stringCollectionDeserializer1, stringCollectionDeserializer0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Class<String> class0 = String.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(simpleType0);
      StringCollectionDeserializer stringCollectionDeserializer0 = new StringCollectionDeserializer(simpleType0, (JsonDeserializer<?>) null, valueInstantiator_Base0);
      StringCollectionDeserializer stringCollectionDeserializer1 = new StringCollectionDeserializer(simpleType0, stringCollectionDeserializer0, valueInstantiator_Base0);
      boolean boolean0 = stringCollectionDeserializer1.isCachable();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Class<String> class0 = String.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(simpleType0);
      StringCollectionDeserializer stringCollectionDeserializer0 = new StringCollectionDeserializer(simpleType0, (JsonDeserializer<?>) null, valueInstantiator_Base0);
      StringCollectionDeserializer stringCollectionDeserializer1 = new StringCollectionDeserializer(simpleType0, valueInstantiator_Base0, stringCollectionDeserializer0, (JsonDeserializer<?>) null, (NullValueProvider) null, (Boolean) null);
      boolean boolean0 = stringCollectionDeserializer1.isCachable();
      assertFalse(boolean0);
      assertTrue(stringCollectionDeserializer0.isCachable());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      StringCollectionDeserializer stringCollectionDeserializer0 = new StringCollectionDeserializer((JavaType) null, (JsonDeserializer<?>) null, (ValueInstantiator) null);
      // Undeclared exception!
      try { 
        stringCollectionDeserializer0.createContextual((DeserializationContext) null, (BeanProperty) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.StringCollectionDeserializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Class<Object> class0 = Object.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(simpleType0);
      JsonDeserializer<Integer> jsonDeserializer0 = (JsonDeserializer<Integer>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      StringCollectionDeserializer stringCollectionDeserializer0 = new StringCollectionDeserializer(simpleType0, jsonDeserializer0, valueInstantiator_Base0);
      assertFalse(stringCollectionDeserializer0.isCachable());
      
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      BeanProperty.Bogus beanProperty_Bogus0 = new BeanProperty.Bogus();
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      JsonDeserializer<?> jsonDeserializer1 = stringCollectionDeserializer0.createContextual(defaultDeserializationContext_Impl0, beanProperty_Bogus0);
      assertSame(stringCollectionDeserializer0, jsonDeserializer1);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Class<String> class0 = String.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(simpleType0);
      StringCollectionDeserializer stringCollectionDeserializer0 = new StringCollectionDeserializer(simpleType0, (JsonDeserializer<?>) null, valueInstantiator_Base0);
      Boolean boolean0 = Boolean.TRUE;
      JsonFactory jsonFactory0 = new JsonFactory();
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      JsonParser jsonParser0 = jsonFactory0.createNonBlockingByteArrayParser();
      StringCollectionDeserializer stringCollectionDeserializer1 = new StringCollectionDeserializer(simpleType0, valueInstantiator_Base0, stringCollectionDeserializer0, stringCollectionDeserializer0, (NullValueProvider) null, boolean0);
      // Undeclared exception!
      try { 
        stringCollectionDeserializer1.deserialize(jsonParser0, (DeserializationContext) defaultDeserializationContext_Impl0, (Collection<String>) null);
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
      RoundingMode roundingMode0 = RoundingMode.UP;
      JsonDeserializer<RoundingMode> jsonDeserializer0 = (JsonDeserializer<RoundingMode>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      doReturn(roundingMode0).when(jsonDeserializer0).deserialize(any(com.fasterxml.jackson.core.JsonParser.class) , any(com.fasterxml.jackson.databind.DeserializationContext.class));
      StringCollectionDeserializer stringCollectionDeserializer0 = new StringCollectionDeserializer((JavaType) null, (ValueInstantiator) null, jsonDeserializer0, jsonDeserializer0, jsonDeserializer0, (Boolean) null);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
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
  public void test12()  throws Throwable  {
      Class<String> class0 = String.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(simpleType0);
      StringCollectionDeserializer stringCollectionDeserializer0 = new StringCollectionDeserializer(simpleType0, (JsonDeserializer<?>) null, valueInstantiator_Base0);
      JsonFactory jsonFactory0 = new JsonFactory();
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      JsonParser jsonParser0 = jsonFactory0.createNonBlockingByteArrayParser();
      ArrayList<String> arrayList0 = new ArrayList<String>();
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      // Undeclared exception!
      try { 
        stringCollectionDeserializer0.deserialize(jsonParser0, (DeserializationContext) defaultDeserializationContext_Impl0, (Collection<String>) arrayList0);
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
      Class<String> class0 = String.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(simpleType0);
      Boolean boolean0 = Boolean.FALSE;
      JsonFactory jsonFactory0 = new JsonFactory();
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      JsonParser jsonParser0 = jsonFactory0.createNonBlockingByteArrayParser();
      ArrayList<String> arrayList0 = new ArrayList<String>();
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      StringCollectionDeserializer stringCollectionDeserializer0 = new StringCollectionDeserializer(simpleType0, (JsonDeserializer<?>) null, valueInstantiator_Base0);
      StringCollectionDeserializer stringCollectionDeserializer1 = stringCollectionDeserializer0.withResolved((JsonDeserializer<?>) null, (JsonDeserializer<?>) null, (NullValueProvider) null, boolean0);
      // Undeclared exception!
      try { 
        stringCollectionDeserializer1.deserialize(jsonParser0, (DeserializationContext) defaultDeserializationContext_Impl0, (Collection<String>) arrayList0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Class<String> class0 = String.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(simpleType0);
      Boolean boolean0 = Boolean.TRUE;
      JsonFactory jsonFactory0 = new JsonFactory();
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      JsonParser jsonParser0 = jsonFactory0.createNonBlockingByteArrayParser();
      ArrayList<String> arrayList0 = new ArrayList<String>();
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      StringCollectionDeserializer stringCollectionDeserializer0 = new StringCollectionDeserializer(simpleType0, (JsonDeserializer<?>) null, valueInstantiator_Base0);
      StringCollectionDeserializer stringCollectionDeserializer1 = stringCollectionDeserializer0.withResolved((JsonDeserializer<?>) null, (JsonDeserializer<?>) null, (NullValueProvider) null, boolean0);
      // Undeclared exception!
      try { 
        stringCollectionDeserializer1.deserialize(jsonParser0, (DeserializationContext) defaultDeserializationContext_Impl0, (Collection<String>) arrayList0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }
}