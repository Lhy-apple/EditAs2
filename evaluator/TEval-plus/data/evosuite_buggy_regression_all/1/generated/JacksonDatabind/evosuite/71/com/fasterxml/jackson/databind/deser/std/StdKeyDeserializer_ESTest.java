/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 13:40:24 GMT 2023
 */

package com.fasterxml.jackson.databind.deser.std;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonToken;
import com.fasterxml.jackson.databind.DeserializationConfig;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.InjectableValues;
import com.fasterxml.jackson.databind.JsonDeserializer;
import com.fasterxml.jackson.databind.JsonSerializer;
import com.fasterxml.jackson.databind.cfg.BaseSettings;
import com.fasterxml.jackson.databind.cfg.DeserializerFactoryConfig;
import com.fasterxml.jackson.databind.deser.BeanDeserializerFactory;
import com.fasterxml.jackson.databind.deser.DefaultDeserializationContext;
import com.fasterxml.jackson.databind.deser.std.StdKeyDeserializer;
import com.fasterxml.jackson.databind.deser.std.UUIDDeserializer;
import com.fasterxml.jackson.databind.introspect.AnnotatedMethod;
import com.fasterxml.jackson.databind.introspect.SimpleMixInResolver;
import com.fasterxml.jackson.databind.jsontype.impl.StdSubtypeResolver;
import com.fasterxml.jackson.databind.util.EnumResolver;
import com.fasterxml.jackson.databind.util.RootNameLookup;
import java.io.IOException;
import java.lang.reflect.Constructor;
import java.lang.reflect.Method;
import java.net.URI;
import java.net.URL;
import java.util.Calendar;
import java.util.Currency;
import java.util.Date;
import java.util.Locale;
import java.util.UUID;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.evosuite.runtime.mock.java.net.MockURL;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class StdKeyDeserializer_ESTest extends StdKeyDeserializer_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Class<Object> class0 = Object.class;
      UUIDDeserializer uUIDDeserializer0 = new UUIDDeserializer();
      StdKeyDeserializer.DelegatingKD stdKeyDeserializer_DelegatingKD0 = new StdKeyDeserializer.DelegatingKD(class0, uUIDDeserializer0);
      Class<?> class1 = stdKeyDeserializer_DelegatingKD0.getKeyClass();
      assertFalse(class1.isInterface());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Class<Byte> class0 = Byte.class;
      StdKeyDeserializer stdKeyDeserializer0 = StdKeyDeserializer.forType(class0);
      // Undeclared exception!
      try { 
        stdKeyDeserializer0.deserializeKey("2143", (DeserializationContext) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.StdKeyDeserializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Class<Long> class0 = Long.class;
      StdKeyDeserializer stdKeyDeserializer0 = StdKeyDeserializer.forType(class0);
      assertNotNull(stdKeyDeserializer0);
      
      Object object0 = stdKeyDeserializer0.deserializeKey("-5", (DeserializationContext) null);
      assertEquals((-5L), object0);
      assertNotNull(object0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Class<Float> class0 = Float.class;
      StdKeyDeserializer stdKeyDeserializer0 = StdKeyDeserializer.forType(class0);
      assertNotNull(stdKeyDeserializer0);
      
      Object object0 = stdKeyDeserializer0.deserializeKey("-5", (DeserializationContext) null);
      assertEquals((-5.0F), object0);
      assertNotNull(object0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Class<Double> class0 = Double.class;
      StdKeyDeserializer stdKeyDeserializer0 = StdKeyDeserializer.forType(class0);
      Class<?> class1 = stdKeyDeserializer0.getKeyClass();
      assertEquals("class java.lang.Double", class1.toString());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Class<UUID> class0 = UUID.class;
      StdKeyDeserializer.StringKD stdKeyDeserializer_StringKD0 = StdKeyDeserializer.StringKD.forType(class0);
      Object object0 = stdKeyDeserializer_StringKD0.deserializeKey((String) null, (DeserializationContext) null);
      assertNull(object0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      StdKeyDeserializer.StringCtorKeyDeserializer stdKeyDeserializer_StringCtorKeyDeserializer0 = null;
      try {
        stdKeyDeserializer_StringCtorKeyDeserializer0 = new StdKeyDeserializer.StringCtorKeyDeserializer((Constructor<?>) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.StdKeyDeserializer$StringCtorKeyDeserializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      StdKeyDeserializer.StringFactoryKeyDeserializer stdKeyDeserializer_StringFactoryKeyDeserializer0 = null;
      try {
        stdKeyDeserializer_StringFactoryKeyDeserializer0 = new StdKeyDeserializer.StringFactoryKeyDeserializer((Method) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.StdKeyDeserializer$StringFactoryKeyDeserializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Class<String> class0 = String.class;
      StdKeyDeserializer stdKeyDeserializer0 = StdKeyDeserializer.forType(class0);
      assertEquals(10, StdKeyDeserializer.TYPE_DATE);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Class<Object> class0 = Object.class;
      StdKeyDeserializer stdKeyDeserializer0 = StdKeyDeserializer.forType(class0);
      assertEquals(16, StdKeyDeserializer.TYPE_CURRENCY);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Class<UUID> class0 = UUID.class;
      StdKeyDeserializer stdKeyDeserializer0 = StdKeyDeserializer.forType(class0);
      Object object0 = stdKeyDeserializer0.deserializeKey("-5", (DeserializationContext) null);
      assertEquals("00000000-0100-4000-8200-000003000000", object0.toString());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Class<Integer> class0 = Integer.class;
      StdKeyDeserializer stdKeyDeserializer0 = StdKeyDeserializer.forType(class0);
      assertNotNull(stdKeyDeserializer0);
      
      Object object0 = stdKeyDeserializer0.deserializeKey("-7", (DeserializationContext) null);
      assertEquals((-7), object0);
      assertNotNull(object0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Class<Date> class0 = Date.class;
      StdKeyDeserializer stdKeyDeserializer0 = StdKeyDeserializer.forType(class0);
      // Undeclared exception!
      try { 
        stdKeyDeserializer0.deserializeKey("-5", (DeserializationContext) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.StdKeyDeserializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Class<Calendar> class0 = Calendar.class;
      StdKeyDeserializer stdKeyDeserializer0 = StdKeyDeserializer.forType(class0);
      // Undeclared exception!
      try { 
        stdKeyDeserializer0.deserializeKey("com.fasterxml.jackson.databind.type.ClassStack", (DeserializationContext) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.StdKeyDeserializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Class<Boolean> class0 = Boolean.class;
      StdKeyDeserializer stdKeyDeserializer0 = StdKeyDeserializer.forType(class0);
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory(deserializerFactoryConfig0);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Object object0 = stdKeyDeserializer0.deserializeKey("false", defaultDeserializationContext_Impl0);
      assertEquals(false, object0);
      assertNotNull(object0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Class<Character> class0 = Character.class;
      StdKeyDeserializer stdKeyDeserializer0 = StdKeyDeserializer.forType(class0);
      try { 
        stdKeyDeserializer0._parse("@,a-xD0w3FJmC4,%", (DeserializationContext) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.StdKeyDeserializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Class<Short> class0 = Short.class;
      StdKeyDeserializer stdKeyDeserializer0 = StdKeyDeserializer.forType(class0);
      assertNotNull(stdKeyDeserializer0);
      
      Object object0 = stdKeyDeserializer0._parse("-5", (DeserializationContext) null);
      assertEquals((short) (-5), object0);
      assertNotNull(object0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Class<Locale> class0 = Locale.class;
      StdKeyDeserializer stdKeyDeserializer0 = StdKeyDeserializer.forType(class0);
      Locale locale0 = (Locale)stdKeyDeserializer0.deserializeKey("", (DeserializationContext) null);
      assertEquals("", locale0.getLanguage());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Class<URI> class0 = URI.class;
      StdKeyDeserializer stdKeyDeserializer0 = StdKeyDeserializer.forType(class0);
      // Undeclared exception!
      try { 
        stdKeyDeserializer0.deserializeKey("v\"", (DeserializationContext) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.StdKeyDeserializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Class<URL> class0 = URL.class;
      StdKeyDeserializer stdKeyDeserializer0 = StdKeyDeserializer.forType(class0);
      // Undeclared exception!
      try { 
        stdKeyDeserializer0.deserializeKey("-5", (DeserializationContext) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.StdKeyDeserializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Class<JsonSerializer> class0 = JsonSerializer.class;
      StdKeyDeserializer stdKeyDeserializer0 = StdKeyDeserializer.forType(class0);
      assertNull(stdKeyDeserializer0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Class<Currency> class0 = Currency.class;
      StdKeyDeserializer stdKeyDeserializer0 = StdKeyDeserializer.forType(class0);
      // Undeclared exception!
      try { 
        stdKeyDeserializer0.deserializeKey("/{3KpG", (DeserializationContext) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.StdKeyDeserializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Class<Byte> class0 = Byte.class;
      StdKeyDeserializer stdKeyDeserializer0 = StdKeyDeserializer.forType(class0);
      Object object0 = stdKeyDeserializer0.deserializeKey((String) null, (DeserializationContext) null);
      assertNull(object0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Class<Long> class0 = Long.TYPE;
      StdKeyDeserializer stdKeyDeserializer0 = new StdKeyDeserializer(248, class0);
      // Undeclared exception!
      try { 
        stdKeyDeserializer0.deserializeKey("", (DeserializationContext) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.StdKeyDeserializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      StdSubtypeResolver stdSubtypeResolver0 = new StdSubtypeResolver();
      RootNameLookup rootNameLookup0 = new RootNameLookup();
      DeserializationConfig deserializationConfig0 = new DeserializationConfig((BaseSettings) null, stdSubtypeResolver0, (SimpleMixInResolver) null, rootNameLookup0);
      InjectableValues.Std injectableValues_Std0 = new InjectableValues.Std();
      DefaultDeserializationContext defaultDeserializationContext0 = defaultDeserializationContext_Impl0.createInstance(deserializationConfig0, (JsonParser) null, injectableValues_Std0);
      Class<JsonToken> class0 = JsonToken.class;
      StdKeyDeserializer stdKeyDeserializer0 = new StdKeyDeserializer(1275, class0);
      try { 
        stdKeyDeserializer0.deserializeKey("mspJr/;)=?u@SX", defaultDeserializationContext0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Can not construct Map key of type com.fasterxml.jackson.core.JsonToken from String (\"mspJr/;)=?u@SX\"): not a valid representation
         //
         verifyException("com.fasterxml.jackson.databind.exc.InvalidFormatException", e);
      }
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      Class<Character> class0 = Character.class;
      StdKeyDeserializer stdKeyDeserializer0 = StdKeyDeserializer.forType(class0);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Object object0 = stdKeyDeserializer0.deserializeKey("-", defaultDeserializationContext_Impl0);
      assertEquals('-', object0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Class<Double> class0 = Double.class;
      StdKeyDeserializer stdKeyDeserializer0 = StdKeyDeserializer.forType(class0);
      // Undeclared exception!
      try { 
        stdKeyDeserializer0.deserializeKey("g{A", (DeserializationContext) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.StdKeyDeserializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Class<Character> class0 = Character.TYPE;
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      StdKeyDeserializer stdKeyDeserializer0 = new StdKeyDeserializer(15, class0);
      try { 
        stdKeyDeserializer0.deserializeKey("9", defaultDeserializationContext_Impl0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Can not construct Map key of type char from String (\"9\"): not a valid representation: Can not construct Map key of type char from String (\"9\"): unable to parse key as Class
         //
         verifyException("com.fasterxml.jackson.databind.exc.InvalidFormatException", e);
      }
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Class<Boolean> class0 = Boolean.class;
      StdKeyDeserializer stdKeyDeserializer0 = StdKeyDeserializer.forType(class0);
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory(deserializerFactoryConfig0);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Object object0 = stdKeyDeserializer0._parse("true", defaultDeserializationContext_Impl0);
      assertEquals(true, object0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Class<Boolean> class0 = Boolean.class;
      StdKeyDeserializer stdKeyDeserializer0 = StdKeyDeserializer.forType(class0);
      // Undeclared exception!
      try { 
        stdKeyDeserializer0.deserializeKey("x5ffk;ewOS^%$Vt =", (DeserializationContext) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.StdKeyDeserializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      Class<Byte> class0 = Byte.class;
      StdKeyDeserializer stdKeyDeserializer0 = StdKeyDeserializer.forType(class0);
      Object object0 = stdKeyDeserializer0.deserializeKey("-5", (DeserializationContext) null);
      assertEquals((byte) (-5), object0);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Class<Object> class0 = Object.class;
      UUIDDeserializer uUIDDeserializer0 = new UUIDDeserializer();
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      StdKeyDeserializer.DelegatingKD stdKeyDeserializer_DelegatingKD0 = new StdKeyDeserializer.DelegatingKD(class0, uUIDDeserializer0);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Object object0 = stdKeyDeserializer_DelegatingKD0.deserializeKey((String) null, defaultDeserializationContext_Impl0);
      assertNull(object0);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      Class<Long> class0 = Long.TYPE;
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      JsonDeserializer<URI> jsonDeserializer0 = (JsonDeserializer<URI>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      doReturn((URI) null).when(jsonDeserializer0).deserialize(any(com.fasterxml.jackson.core.JsonParser.class) , any(com.fasterxml.jackson.databind.DeserializationContext.class));
      StdKeyDeserializer.DelegatingKD stdKeyDeserializer_DelegatingKD0 = new StdKeyDeserializer.DelegatingKD(class0, jsonDeserializer0);
      try { 
        stdKeyDeserializer_DelegatingKD0.deserializeKey("[~V$V%", defaultDeserializationContext_Impl0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Can not construct Map key of type long from String (\"[~V$V%\"): not a valid representation
         //
         verifyException("com.fasterxml.jackson.databind.exc.InvalidFormatException", e);
      }
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Class<Long> class0 = Long.class;
      URL uRL0 = MockURL.getFtpExample();
      URI uRI0 = MockURL.toURI(uRL0);
      JsonDeserializer<URI> jsonDeserializer0 = (JsonDeserializer<URI>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      doReturn(uRI0).when(jsonDeserializer0).deserialize(any(com.fasterxml.jackson.core.JsonParser.class) , any(com.fasterxml.jackson.databind.DeserializationContext.class));
      StdKeyDeserializer.DelegatingKD stdKeyDeserializer_DelegatingKD0 = new StdKeyDeserializer.DelegatingKD(class0, jsonDeserializer0);
      Object object0 = stdKeyDeserializer_DelegatingKD0.deserializeKey("[~V$V%", defaultDeserializationContext_Impl0);
      assertSame(uRI0, object0);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      Class<JsonToken> class0 = JsonToken.class;
      EnumResolver enumResolver0 = EnumResolver.constructUnsafeUsingToString(class0);
      StdKeyDeserializer.EnumKD stdKeyDeserializer_EnumKD0 = new StdKeyDeserializer.EnumKD(enumResolver0, (AnnotatedMethod) null);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      StdSubtypeResolver stdSubtypeResolver0 = new StdSubtypeResolver();
      RootNameLookup rootNameLookup0 = new RootNameLookup();
      DeserializationConfig deserializationConfig0 = new DeserializationConfig((BaseSettings) null, stdSubtypeResolver0, (SimpleMixInResolver) null, rootNameLookup0);
      InjectableValues.Std injectableValues_Std0 = new InjectableValues.Std();
      DefaultDeserializationContext defaultDeserializationContext0 = defaultDeserializationContext_Impl0.createInstance(deserializationConfig0, (JsonParser) null, injectableValues_Std0);
      try { 
        stdKeyDeserializer_EnumKD0.deserializeKey(" bytes", defaultDeserializationContext0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Can not construct Map key of type com.fasterxml.jackson.core.JsonToken from String (\" bytes\"): not a valid representation: Can not construct Map key of type com.fasterxml.jackson.core.JsonToken from String (\" bytes\"): not one of values excepted for Enum class: [VALUE_NULL, START_OBJECT, VALUE_FALSE, VALUE_TRUE, VALUE_NUMBER_INT, VALUE_STRING, VALUE_EMBEDDED_OBJECT, END_ARRAY, FIELD_NAME, END_OBJECT, VALUE_NUMBER_FLOAT, START_ARRAY, NOT_AVAILABLE]
         //
         verifyException("com.fasterxml.jackson.databind.exc.InvalidFormatException", e);
      }
  }
}