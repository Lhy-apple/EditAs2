/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 17:44:48 GMT 2023
 */

package com.fasterxml.jackson.databind.deser.std;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.InjectableValues;
import com.fasterxml.jackson.databind.MapperFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
import com.fasterxml.jackson.databind.deser.BeanDeserializerFactory;
import com.fasterxml.jackson.databind.deser.DefaultDeserializationContext;
import com.fasterxml.jackson.databind.deser.std.FromStringDeserializer;
import com.fasterxml.jackson.databind.deser.std.UUIDDeserializer;
import com.fasterxml.jackson.databind.ser.DefaultSerializerProvider;
import com.fasterxml.jackson.databind.type.ArrayType;
import com.fasterxml.jackson.databind.type.CollectionType;
import java.io.File;
import java.net.Inet4Address;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.net.MalformedURLException;
import java.net.URI;
import java.net.URL;
import java.nio.charset.Charset;
import java.nio.charset.IllegalCharsetNameException;
import java.util.Currency;
import java.util.HashMap;
import java.util.Locale;
import java.util.TimeZone;
import java.util.regex.Pattern;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.io.MockFile;
import org.evosuite.runtime.mock.java.net.MockInetSocketAddress;
import org.junit.runner.RunWith;
import sun.util.calendar.ZoneInfo;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class FromStringDeserializer_ESTest extends FromStringDeserializer_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Class<InetAddress> class0 = InetAddress.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 484);
      JsonFactory jsonFactory0 = new JsonFactory();
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      JsonParser jsonParser0 = jsonFactory0.createParser("JSON");
      // Undeclared exception!
      try { 
        fromStringDeserializer_Std0.deserialize(jsonParser0, (DeserializationContext) defaultDeserializationContext_Impl0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      UUIDDeserializer uUIDDeserializer0 = new UUIDDeserializer();
      // Undeclared exception!
      try { 
        uUIDDeserializer0._deserializeEmbedded((Object) null, (DeserializationContext) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.FromStringDeserializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Class<InetSocketAddress> class0 = InetSocketAddress.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = FromStringDeserializer.findDeserializer(class0);
      Object object0 = fromStringDeserializer_Std0._deserializeFromEmptyString();
      assertNull(object0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Class<?>[] classArray0 = FromStringDeserializer.types();
      assertEquals(13, classArray0.length);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Class<File> class0 = File.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = FromStringDeserializer.findDeserializer(class0);
      assertEquals(5, FromStringDeserializer.Std.STD_JAVA_TYPE);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Class<URL> class0 = URL.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = FromStringDeserializer.findDeserializer(class0);
      assertEquals(2, FromStringDeserializer.Std.STD_URL);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Class<URI> class0 = URI.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = FromStringDeserializer.findDeserializer(class0);
      assertEquals(5, FromStringDeserializer.Std.STD_JAVA_TYPE);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Class<ArrayType> class0 = ArrayType.class;
      JsonFactory jsonFactory0 = new JsonFactory();
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      ObjectMapper objectMapper0 = new ObjectMapper(jsonFactory0, defaultSerializerProvider_Impl0, defaultDeserializationContext_Impl0);
      HashMap<String, Object> hashMap0 = new HashMap<String, Object>();
      InjectableValues.Std injectableValues_Std0 = new InjectableValues.Std(hashMap0);
      ObjectReader objectReader0 = objectMapper0.reader((InjectableValues) injectableValues_Std0);
      ObjectReader objectReader1 = objectReader0.forType(class0);
      assertNotSame(objectReader1, objectReader0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Class<Currency> class0 = Currency.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = FromStringDeserializer.findDeserializer(class0);
      assertFalse(fromStringDeserializer_Std0.isCachable());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Class<Pattern> class0 = Pattern.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = FromStringDeserializer.findDeserializer(class0);
      assertEquals(13, FromStringDeserializer.Std.STD_STRING_BUILDER);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Class<Locale> class0 = Locale.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = FromStringDeserializer.findDeserializer(class0);
      assertEquals(9, FromStringDeserializer.Std.STD_CHARSET);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Class<Charset> class0 = Charset.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = FromStringDeserializer.findDeserializer(class0);
      assertFalse(fromStringDeserializer_Std0.isCachable());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Class<TimeZone> class0 = TimeZone.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = FromStringDeserializer.findDeserializer(class0);
      assertEquals(2, FromStringDeserializer.Std.STD_URL);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Class<InetAddress> class0 = InetAddress.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = FromStringDeserializer.findDeserializer(class0);
      assertEquals(11, FromStringDeserializer.Std.STD_INET_ADDRESS);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Class<ArrayType> class0 = ArrayType.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = FromStringDeserializer.findDeserializer(class0);
      assertNull(fromStringDeserializer_Std0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Class<StringBuilder> class0 = StringBuilder.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = FromStringDeserializer.findDeserializer(class0);
      assertEquals(13, FromStringDeserializer.Std.STD_STRING_BUILDER);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Class<InetSocketAddress> class0 = InetSocketAddress.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 1);
      MockFile mockFile0 = (MockFile)fromStringDeserializer_Std0._deserialize("", (DeserializationContext) null);
      assertTrue(mockFile0.canExecute());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Class<ArrayType> class0 = ArrayType.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 12);
      Object object0 = fromStringDeserializer_Std0._deserialize("[xhnP#:ZQ]Wgjb2B", (DeserializationContext) null);
      assertEquals("/200.42.42.0:0", object0.toString());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Class<InetAddress> class0 = InetAddress.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 2);
      try { 
        fromStringDeserializer_Std0._deserialize("[xhnP#:ZQ]Wgjb2B", (DeserializationContext) null);
        fail("Expecting exception: MalformedURLException");
      
      } catch(MalformedURLException e) {
         //
         // no protocol: [xhnP#:ZQ]Wgjb2B
         //
         verifyException("java.net.URL", e);
      }
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Class<InetSocketAddress> class0 = InetSocketAddress.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 3);
      // Undeclared exception!
      try { 
        fromStringDeserializer_Std0._deserialize("[xhnP#:ZQ]Wgjb2B", (DeserializationContext) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Illegal character in path at index 0: [xhnP#:ZQ]Wgjb2B
         //
         verifyException("java.net.URI", e);
      }
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Class<Locale> class0 = Locale.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 4);
      // Undeclared exception!
      try { 
        fromStringDeserializer_Std0._deserialize("[[Don't know how to convert embedded Object of type %s into %s", (DeserializationContext) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.FromStringDeserializer$Std", e);
      }
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Class<ArrayType> class0 = ArrayType.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 5);
      // Undeclared exception!
      try { 
        fromStringDeserializer_Std0._deserialize("xhnP#:ZQ]Wgjb2B", (DeserializationContext) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.FromStringDeserializer$Std", e);
      }
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Class<Object> class0 = Object.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 6);
      // Undeclared exception!
      try { 
        fromStringDeserializer_Std0._deserialize("V6x7z{jFI", (DeserializationContext) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("java.util.Currency", e);
      }
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Class<URL> class0 = URL.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 7);
      Object object0 = fromStringDeserializer_Std0._deserialize("", (DeserializationContext) null);
      assertEquals("", object0.toString());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Class<ArrayType> class0 = ArrayType.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 9);
      // Undeclared exception!
      try { 
        fromStringDeserializer_Std0._deserialize("[xhnP#:ZQ]Wgjb2B", (DeserializationContext) null);
        fail("Expecting exception: IllegalCharsetNameException");
      
      } catch(IllegalCharsetNameException e) {
         //
         // [xhnP#:ZQ]Wgjb2B
         //
         verifyException("java.nio.charset.Charset", e);
      }
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Class<ArrayType> class0 = ArrayType.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 10);
      ZoneInfo zoneInfo0 = (ZoneInfo)fromStringDeserializer_Std0._deserialize("P", (DeserializationContext) null);
      assertEquals("GMT", zoneInfo0.getID());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Class<InetSocketAddress> class0 = InetSocketAddress.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 11);
      Inet4Address inet4Address0 = (Inet4Address)fromStringDeserializer_Std0._deserialize("[xhnP#:ZQ]Wgjb2B", (DeserializationContext) null);
      assertFalse(inet4Address0.isMCNodeLocal());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Class<ArrayType> class0 = ArrayType.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 13);
      Object object0 = fromStringDeserializer_Std0._deserialize("(", (DeserializationContext) null);
      assertEquals("(", object0.toString());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Class<ArrayType> class0 = ArrayType.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, (-3873));
      // Undeclared exception!
      try { 
        fromStringDeserializer_Std0._deserialize("(JK#j/", (DeserializationContext) null);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // Internal error: this code path should never get executed
         //
         verifyException("com.fasterxml.jackson.core.util.VersionUtil", e);
      }
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Class<ArrayType> class0 = ArrayType.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 8);
      Locale locale0 = (Locale)fromStringDeserializer_Std0._deserialize("N", (DeserializationContext) null);
      assertEquals("n", locale0.getLanguage());
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      DeserializationContext deserializationContext0 = objectMapper0.getDeserializationContext();
      Class<CollectionType> class0 = CollectionType.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 8);
      Object object0 = fromStringDeserializer_Std0._deserialize(" You can disable the check via `JsonFactory.Feature.FAIL_ON_SYMBOL_HASH_OVERFLOW`", deserializationContext0);
      assertEquals(" you can disable the check via `jsonfactory.feature.fail_ON_SYMBOL_HASH_OVERFLOW`", object0.toString());
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Class<ArrayType> class0 = ArrayType.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 12);
      MockInetSocketAddress mockInetSocketAddress0 = (MockInetSocketAddress)fromStringDeserializer_Std0._deserialize("com.fasterxml.jackson.databind.ser.BeanPropertyWriter", (DeserializationContext) null);
      assertFalse(mockInetSocketAddress0.isUnresolved());
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      Class<ArrayType> class0 = ArrayType.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 12);
      // Undeclared exception!
      try { 
        fromStringDeserializer_Std0._deserialize("[", (DeserializationContext) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.FromStringDeserializer$Std", e);
      }
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Class<ArrayType> class0 = ArrayType.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 12);
      // Undeclared exception!
      try { 
        fromStringDeserializer_Std0._deserialize("xhnP#:ZQ]Wjb2B", (DeserializationContext) null);
        fail("Expecting exception: NumberFormatException");
      
      } catch(NumberFormatException e) {
         //
         // For input string: \"ZQ]Wjb2B\"
         //
         verifyException("java.lang.NumberFormatException", e);
      }
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      Class<InetSocketAddress> class0 = InetSocketAddress.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 3);
      URI uRI0 = (URI)fromStringDeserializer_Std0._deserializeFromEmptyString();
      assertEquals((-1), uRI0.getPort());
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      Class<MapperFeature> class0 = MapperFeature.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 8);
      Locale locale0 = (Locale)fromStringDeserializer_Std0._deserializeFromEmptyString();
      assertEquals("", locale0.getISO3Language());
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      Class<File> class0 = File.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 13);
      Object object0 = fromStringDeserializer_Std0._deserializeFromEmptyString();
      assertEquals("", object0.toString());
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      Class<ArrayType> class0 = ArrayType.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 8);
      Object object0 = fromStringDeserializer_Std0._deserialize("F<U1U5-V", (DeserializationContext) null);
      assertEquals("f<u1u5_V", object0.toString());
  }
}