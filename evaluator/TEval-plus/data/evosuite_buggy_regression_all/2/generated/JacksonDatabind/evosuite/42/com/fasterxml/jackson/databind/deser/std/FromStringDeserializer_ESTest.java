/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 15:00:05 GMT 2023
 */

package com.fasterxml.jackson.databind.deser.std;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.cfg.DeserializerFactoryConfig;
import com.fasterxml.jackson.databind.deser.BeanDeserializerFactory;
import com.fasterxml.jackson.databind.deser.DefaultDeserializationContext;
import com.fasterxml.jackson.databind.deser.std.FromStringDeserializer;
import com.fasterxml.jackson.databind.deser.std.UUIDDeserializer;
import com.fasterxml.jackson.databind.type.MapLikeType;
import java.io.File;
import java.io.IOException;
import java.net.Inet4Address;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.net.MalformedURLException;
import java.net.URI;
import java.net.URL;
import java.nio.charset.Charset;
import java.nio.charset.IllegalCharsetNameException;
import java.util.Currency;
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
  public void test01()  throws Throwable  {
      Class<InetSocketAddress> class0 = InetSocketAddress.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = FromStringDeserializer.findDeserializer(class0);
      Object object0 = fromStringDeserializer_Std0._deserializeFromEmptyString();
      assertNull(object0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Class<?>[] classArray0 = FromStringDeserializer.types();
      assertEquals(12, classArray0.length);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Class<File> class0 = File.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = FromStringDeserializer.findDeserializer(class0);
      assertEquals(10, FromStringDeserializer.Std.STD_TIME_ZONE);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Class<URL> class0 = URL.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = FromStringDeserializer.findDeserializer(class0);
      assertEquals(12, FromStringDeserializer.Std.STD_INET_SOCKET_ADDRESS);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Class<URI> class0 = URI.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = FromStringDeserializer.findDeserializer(class0);
      URI uRI0 = (URI)fromStringDeserializer_Std0._deserializeFromEmptyString();
      assertNull(uRI0.getRawUserInfo());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Class<Currency> class0 = Currency.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = FromStringDeserializer.findDeserializer(class0);
      assertEquals(4, FromStringDeserializer.Std.STD_CLASS);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Class<Pattern> class0 = Pattern.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = FromStringDeserializer.findDeserializer(class0);
      assertEquals(7, FromStringDeserializer.Std.STD_PATTERN);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Class<Locale> class0 = Locale.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = FromStringDeserializer.findDeserializer(class0);
      assertEquals(7, FromStringDeserializer.Std.STD_PATTERN);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Class<Charset> class0 = Charset.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = FromStringDeserializer.findDeserializer(class0);
      assertEquals(6, FromStringDeserializer.Std.STD_CURRENCY);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Class<TimeZone> class0 = TimeZone.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = FromStringDeserializer.findDeserializer(class0);
      assertEquals(9, FromStringDeserializer.Std.STD_CHARSET);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Class<InetAddress> class0 = InetAddress.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = FromStringDeserializer.findDeserializer(class0);
      assertEquals(7, FromStringDeserializer.Std.STD_PATTERN);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Class<String> class0 = String.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = FromStringDeserializer.findDeserializer(class0);
      assertNull(fromStringDeserializer_Std0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      JsonFactory jsonFactory0 = new JsonFactory();
      JsonParser jsonParser0 = jsonFactory0.createParser((char[]) null, 10, 10);
      Class<Integer> class0 = Integer.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 10);
      // Undeclared exception!
      try { 
        fromStringDeserializer_Std0.deserialize(jsonParser0, (DeserializationContext) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.FromStringDeserializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Class<Charset> class0 = Charset.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 1);
      MockFile mockFile0 = (MockFile)fromStringDeserializer_Std0._deserialize("@P|DMt", (DeserializationContext) null);
      assertEquals(0L, mockFile0.length());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Class<InetSocketAddress> class0 = InetSocketAddress.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 12);
      try { 
        fromStringDeserializer_Std0._deserialize("[' value but there was more than a single value in the aray", (DeserializationContext) null);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Bracketed IPv6 address must contain closing bracket
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.FromStringDeserializer$Std", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Class<String> class0 = String.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 2);
      try { 
        fromStringDeserializer_Std0._deserialize("[", (DeserializationContext) null);
        fail("Expecting exception: MalformedURLException");
      
      } catch(MalformedURLException e) {
         //
         // no protocol: [
         //
         verifyException("java.net.URL", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Class<Currency> class0 = Currency.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 3);
      URI uRI0 = (URI)fromStringDeserializer_Std0._deserialize("_0", (DeserializationContext) null);
      assertNull(uRI0.getRawAuthority());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Class<String> class0 = String.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 4);
      // Undeclared exception!
      try { 
        fromStringDeserializer_Std0._deserialize("\u00079hvyG", (DeserializationContext) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.FromStringDeserializer$Std", e);
      }
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Class<String> class0 = String.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 5);
      // Undeclared exception!
      try { 
        fromStringDeserializer_Std0._deserialize("JsonSerializer of type ", (DeserializationContext) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.FromStringDeserializer$Std", e);
      }
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Class<URI> class0 = URI.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 6);
      // Undeclared exception!
      try { 
        fromStringDeserializer_Std0._deserialize("' value but there was more than a single value in the array", (DeserializationContext) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("java.util.Currency", e);
      }
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Class<Integer> class0 = Integer.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 7);
      Object object0 = fromStringDeserializer_Std0._deserialize("KB:y::$4'K}sj", (DeserializationContext) null);
      assertEquals("KB:y::$4'K}sj", object0.toString());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Class<JsonFactory.Feature> class0 = JsonFactory.Feature.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 8);
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory(deserializerFactoryConfig0);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Object object0 = fromStringDeserializer_Std0._deserialize("/_M$", defaultDeserializationContext_Impl0);
      assertEquals("/_M$", object0.toString());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Class<JsonFactory.Feature> class0 = JsonFactory.Feature.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 9);
      // Undeclared exception!
      try { 
        fromStringDeserializer_Std0._deserialize("not a valid textual representation", (DeserializationContext) null);
        fail("Expecting exception: IllegalCharsetNameException");
      
      } catch(IllegalCharsetNameException e) {
         //
         // not a valid textual representation
         //
         verifyException("java.nio.charset.Charset", e);
      }
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Class<Object> class0 = Object.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 10);
      ZoneInfo zoneInfo0 = (ZoneInfo)fromStringDeserializer_Std0._deserialize("[\"c N<*Ag]zy", (DeserializationContext) null);
      assertEquals("GMT", zoneInfo0.getID());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Class<InetSocketAddress> class0 = InetSocketAddress.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 11);
      Inet4Address inet4Address0 = (Inet4Address)fromStringDeserializer_Std0._deserialize("&{k9PV#\"ka0Bbw)7/", (DeserializationContext) null);
      assertFalse(inet4Address0.isMCSiteLocal());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Class<String> class0 = String.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 0);
      // Undeclared exception!
      try { 
        fromStringDeserializer_Std0._deserialize("KB:y::$4'K}sj", (DeserializationContext) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.FromStringDeserializer$Std", e);
      }
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Class<InetSocketAddress> class0 = InetSocketAddress.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 8);
      Locale locale0 = (Locale)fromStringDeserializer_Std0._deserialize("HlBV@f|HwXJ&+q.K", (DeserializationContext) null);
      assertEquals("", locale0.getCountry());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Class<MapLikeType> class0 = MapLikeType.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 8);
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory(deserializerFactoryConfig0);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Object object0 = fromStringDeserializer_Std0._deserialize("WRITE_SINGLE_ELEM_ARRAYS_UNWRAPPED", defaultDeserializationContext_Impl0);
      assertEquals("write_SINGLE_ELEM_ARRAYS_UNWRAPPED", object0.toString());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Class<String> class0 = String.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 12);
      // Undeclared exception!
      try { 
        fromStringDeserializer_Std0._deserialize("Internal error: SimpleType.narrowContentsBy() should never be called", (DeserializationContext) null);
        fail("Expecting exception: NumberFormatException");
      
      } catch(NumberFormatException e) {
         //
         // For input string: \" SimpleType.narrowContentsBy() should never be called\"
         //
         verifyException("java.lang.NumberFormatException", e);
      }
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      Class<Charset> class0 = Charset.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 12);
      MockInetSocketAddress mockInetSocketAddress0 = (MockInetSocketAddress)fromStringDeserializer_Std0._deserialize("[\"c N<*Ag]zy", (DeserializationContext) null);
      assertEquals("200.42.42.0", mockInetSocketAddress0.getHostString());
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Class<JsonFactory.Feature> class0 = JsonFactory.Feature.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 12);
      Object object0 = fromStringDeserializer_Std0._deserialize("_tO>3banw.;K5", (DeserializationContext) null);
      assertEquals("/200.42.42.0:0", object0.toString());
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      Class<TimeZone> class0 = TimeZone.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 12);
      MockInetSocketAddress mockInetSocketAddress0 = (MockInetSocketAddress)fromStringDeserializer_Std0._deserialize("KB:y::$4'K}sj", (DeserializationContext) null);
      assertEquals(0, mockInetSocketAddress0.getPort());
  }
}
