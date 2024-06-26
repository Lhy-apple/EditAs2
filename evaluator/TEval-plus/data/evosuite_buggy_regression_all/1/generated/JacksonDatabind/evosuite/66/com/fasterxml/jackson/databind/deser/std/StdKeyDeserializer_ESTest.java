/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 13:39:53 GMT 2023
 */

package com.fasterxml.jackson.databind.deser.std;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.databind.AnnotationIntrospector;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JsonDeserializer;
import com.fasterxml.jackson.databind.deser.BeanDeserializerFactory;
import com.fasterxml.jackson.databind.deser.DefaultDeserializationContext;
import com.fasterxml.jackson.databind.deser.std.FromStringDeserializer;
import com.fasterxml.jackson.databind.deser.std.StdKeyDeserializer;
import com.fasterxml.jackson.databind.introspect.AnnotatedMethod;
import com.fasterxml.jackson.databind.util.EnumResolver;
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
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class StdKeyDeserializer_ESTest extends StdKeyDeserializer_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Class<URI> class0 = URI.class;
      StdKeyDeserializer stdKeyDeserializer0 = StdKeyDeserializer.forType(class0);
      URI uRI0 = (URI)stdKeyDeserializer0.deserializeKey("com.fasterxml.jackson.databind.deser.std.StdKeyDeserializer$EnumKD", (DeserializationContext) null);
      assertEquals((-1), uRI0.getPort());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Class<Long> class0 = Long.class;
      StdKeyDeserializer stdKeyDeserializer0 = StdKeyDeserializer.forType(class0);
      // Undeclared exception!
      try { 
        stdKeyDeserializer0.deserializeKey("[Mb6]#O.", (DeserializationContext) null);
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
      Class<Double> class0 = Double.class;
      StdKeyDeserializer stdKeyDeserializer0 = StdKeyDeserializer.forType(class0);
      // Undeclared exception!
      try { 
        stdKeyDeserializer0.deserializeKey("MCxQ,azip+.", (DeserializationContext) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.StdKeyDeserializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Class<Long> class0 = Long.class;
      FromStringDeserializer<Long> fromStringDeserializer0 = (FromStringDeserializer<Long>) mock(FromStringDeserializer.class, new ViolatedAssumptionAnswer());
      StdKeyDeserializer stdKeyDeserializer0 = new StdKeyDeserializer(399, class0, fromStringDeserializer0);
      Class<?> class1 = stdKeyDeserializer0.getKeyClass();
      assertFalse(class1.isInterface());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Class<Calendar> class0 = Calendar.class;
      StdKeyDeserializer.StringKD stdKeyDeserializer_StringKD0 = StdKeyDeserializer.StringKD.forType(class0);
      Object object0 = stdKeyDeserializer_StringKD0.deserializeKey("initCause", (DeserializationContext) null);
      assertEquals("initCause", object0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Class<Locale.Category> class0 = Locale.Category.class;
      AnnotationIntrospector annotationIntrospector0 = AnnotationIntrospector.nopInstance();
      EnumResolver enumResolver0 = EnumResolver.constructUnsafe(class0, annotationIntrospector0);
      StdKeyDeserializer.EnumKD stdKeyDeserializer_EnumKD0 = new StdKeyDeserializer.EnumKD(enumResolver0, (AnnotatedMethod) null);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      // Undeclared exception!
      try { 
        stdKeyDeserializer_EnumKD0.deserializeKey("i0U@c9ofA", defaultDeserializationContext_Impl0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
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
      assertEquals(8, StdKeyDeserializer.TYPE_DOUBLE);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Class<Object> class0 = Object.class;
      StdKeyDeserializer stdKeyDeserializer0 = StdKeyDeserializer.forType(class0);
      assertEquals(13, StdKeyDeserializer.TYPE_URI);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Class<UUID> class0 = UUID.class;
      StdKeyDeserializer stdKeyDeserializer0 = StdKeyDeserializer.forType(class0);
      Object object0 = stdKeyDeserializer0._parse("D_/c<M v", (DeserializationContext) null);
      assertEquals("00000000-0100-4000-8200-000003000000", object0.toString());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Class<Integer> class0 = Integer.class;
      StdKeyDeserializer stdKeyDeserializer0 = StdKeyDeserializer.forType(class0);
      // Undeclared exception!
      try { 
        stdKeyDeserializer0.deserializeKey("com.fasterxml.jackson.databind.deser.std.StdKeyDeserializer$EnumKD", (DeserializationContext) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.StdKeyDeserializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Class<Date> class0 = Date.class;
      StdKeyDeserializer stdKeyDeserializer0 = StdKeyDeserializer.forType(class0);
      // Undeclared exception!
      try { 
        stdKeyDeserializer0.deserializeKey("O|becT8.", (DeserializationContext) null);
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
      Object object0 = stdKeyDeserializer0.deserializeKey((String) null, (DeserializationContext) null);
      assertNull(object0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Class<Boolean> class0 = Boolean.class;
      StdKeyDeserializer stdKeyDeserializer0 = StdKeyDeserializer.forType(class0);
      // Undeclared exception!
      try { 
        stdKeyDeserializer0.deserializeKey("9sTd1'{:", (DeserializationContext) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.StdKeyDeserializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Class<Character> class0 = Character.class;
      StdKeyDeserializer stdKeyDeserializer0 = StdKeyDeserializer.forType(class0);
      // Undeclared exception!
      try { 
        stdKeyDeserializer0.deserializeKey("com.fasterxml.jackson.databind.deser.std.StdKeyDeserializer$EnumKD", (DeserializationContext) null);
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
      Object object0 = stdKeyDeserializer0.deserializeKey("3", (DeserializationContext) null);
      assertEquals((short)3, object0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Class<Float> class0 = Float.class;
      StdKeyDeserializer stdKeyDeserializer0 = StdKeyDeserializer.forType(class0);
      try { 
        stdKeyDeserializer0._parse("not a valid representation", (DeserializationContext) null);
        fail("Expecting exception: NumberFormatException");
      
      } catch(NumberFormatException e) {
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Class<URL> class0 = URL.class;
      StdKeyDeserializer stdKeyDeserializer0 = StdKeyDeserializer.forType(class0);
      // Undeclared exception!
      try { 
        stdKeyDeserializer0.deserializeKey("+", (DeserializationContext) null);
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
      Class<Currency> class0 = Currency.class;
      StdKeyDeserializer stdKeyDeserializer0 = StdKeyDeserializer.forType(class0);
      // Undeclared exception!
      try { 
        stdKeyDeserializer0.deserializeKey("not a valid representation, problem: (%s) %s", (DeserializationContext) null);
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
      Class<Locale> class0 = Locale.class;
      StdKeyDeserializer stdKeyDeserializer0 = StdKeyDeserializer.forType(class0);
      assertEquals(3, StdKeyDeserializer.TYPE_SHORT);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Class<Locale.Category> class0 = Locale.Category.class;
      StdKeyDeserializer stdKeyDeserializer0 = StdKeyDeserializer.forType(class0);
      assertNull(stdKeyDeserializer0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Class<Byte> class0 = Byte.class;
      FromStringDeserializer<Locale> fromStringDeserializer0 = (FromStringDeserializer<Locale>) mock(FromStringDeserializer.class, new ViolatedAssumptionAnswer());
      doReturn((Object) null).when(fromStringDeserializer0)._deserialize(anyString() , any(com.fasterxml.jackson.databind.DeserializationContext.class));
      StdKeyDeserializer stdKeyDeserializer0 = new StdKeyDeserializer(9, class0, fromStringDeserializer0);
      // Undeclared exception!
      try { 
        stdKeyDeserializer0.deserializeKey("0wLkvS-B%c6xDXM3K", (DeserializationContext) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.StdKeyDeserializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Class<Locale.Category> class0 = Locale.Category.class;
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      FromStringDeserializer<URI> fromStringDeserializer0 = (FromStringDeserializer<URI>) mock(FromStringDeserializer.class, new ViolatedAssumptionAnswer());
      doReturn((Object) null).when(fromStringDeserializer0)._deserialize(anyString() , any(com.fasterxml.jackson.databind.DeserializationContext.class));
      StdKeyDeserializer stdKeyDeserializer0 = new StdKeyDeserializer(9, class0, fromStringDeserializer0);
      // Undeclared exception!
      try { 
        stdKeyDeserializer0.deserializeKey("0wLkvS-B%c6xDXM3K", defaultDeserializationContext_Impl0);
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
      Class<Calendar> class0 = Calendar.class;
      StdKeyDeserializer stdKeyDeserializer0 = StdKeyDeserializer.forType(class0);
      // Undeclared exception!
      try { 
        stdKeyDeserializer0.deserializeKey("com.fasterxml.jackson.databind.deser.std.StdKeyDeserializer$EnumKD", (DeserializationContext) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.StdKeyDeserializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Class<Calendar> class0 = Calendar.class;
      StdKeyDeserializer stdKeyDeserializer0 = new StdKeyDeserializer(15, class0);
      // Undeclared exception!
      try { 
        stdKeyDeserializer0.deserializeKey("byte", (DeserializationContext) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.StdKeyDeserializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Class<Byte> class0 = Byte.class;
      StdKeyDeserializer.StringKD stdKeyDeserializer_StringKD0 = StdKeyDeserializer.StringKD.forType(class0);
      try { 
        stdKeyDeserializer_StringKD0._parse("overflow, value can not be represented as 8-bit value", (DeserializationContext) null);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // Internal error: unknown key type class java.lang.Byte
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.StdKeyDeserializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Class<Short> class0 = Short.class;
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      StdKeyDeserializer stdKeyDeserializer0 = new StdKeyDeserializer(1, class0, (FromStringDeserializer<?>) null);
      Object object0 = stdKeyDeserializer0.deserializeKey("true", defaultDeserializationContext_Impl0);
      assertEquals(true, object0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Class<Boolean> class0 = Boolean.class;
      StdKeyDeserializer stdKeyDeserializer0 = StdKeyDeserializer.forType(class0);
      Object object0 = stdKeyDeserializer0._parse("false", (DeserializationContext) null);
      assertEquals(false, object0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Class<Byte> class0 = Byte.class;
      StdKeyDeserializer stdKeyDeserializer0 = StdKeyDeserializer.forType(class0);
      Object object0 = stdKeyDeserializer0.deserializeKey("3", (DeserializationContext) null);
      assertEquals((byte)3, object0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      Class<Character> class0 = Character.class;
      StdKeyDeserializer stdKeyDeserializer0 = StdKeyDeserializer.forType(class0);
      Object object0 = stdKeyDeserializer0.deserializeKey("3", (DeserializationContext) null);
      assertEquals('3', object0);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Class<Object> class0 = Object.class;
      JsonDeserializer<Currency> jsonDeserializer0 = (JsonDeserializer<Currency>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      doReturn((Object) null).when(jsonDeserializer0).deserialize(any(com.fasterxml.jackson.core.JsonParser.class) , any(com.fasterxml.jackson.databind.DeserializationContext.class));
      StdKeyDeserializer.DelegatingKD stdKeyDeserializer_DelegatingKD0 = new StdKeyDeserializer.DelegatingKD(class0, jsonDeserializer0);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      // Undeclared exception!
      try { 
        stdKeyDeserializer_DelegatingKD0.deserializeKey("not a valid representation: %s", defaultDeserializationContext_Impl0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      Class<Object> class0 = Object.class;
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      JsonDeserializer<Double> jsonDeserializer0 = (JsonDeserializer<Double>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      StdKeyDeserializer.DelegatingKD stdKeyDeserializer_DelegatingKD0 = new StdKeyDeserializer.DelegatingKD(class0, jsonDeserializer0);
      Object object0 = stdKeyDeserializer_DelegatingKD0.deserializeKey((String) null, defaultDeserializationContext_Impl0);
      assertNull(object0);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Class<Object> class0 = Object.class;
      Locale locale0 = Locale.PRC;
      Currency.getInstance(locale0);
      JsonDeserializer<Currency> jsonDeserializer0 = (JsonDeserializer<Currency>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      doReturn((Object) null).when(jsonDeserializer0).deserialize(any(com.fasterxml.jackson.core.JsonParser.class) , any(com.fasterxml.jackson.databind.DeserializationContext.class));
      StdKeyDeserializer.DelegatingKD stdKeyDeserializer_DelegatingKD0 = new StdKeyDeserializer.DelegatingKD(class0, jsonDeserializer0);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      // Undeclared exception!
      try { 
        stdKeyDeserializer_DelegatingKD0.deserializeKey("not a valid representation: %s", defaultDeserializationContext_Impl0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }
}
