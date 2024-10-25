/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 13:41:44 GMT 2023
 */

package com.fasterxml.jackson.databind.deser.std;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
import com.fasterxml.jackson.databind.deser.BeanDeserializerFactory;
import com.fasterxml.jackson.databind.deser.DefaultDeserializationContext;
import com.fasterxml.jackson.databind.deser.std.FromStringDeserializer;
import com.fasterxml.jackson.databind.deser.std.UUIDDeserializer;
import com.fasterxml.jackson.databind.type.CollectionLikeType;
import com.fasterxml.jackson.databind.type.SimpleType;
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
import java.util.SimpleTimeZone;
import java.util.TimeZone;
import java.util.regex.Pattern;
import java.util.regex.PatternSyntaxException;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.net.MockInetSocketAddress;
import org.junit.runner.RunWith;
import sun.util.calendar.ZoneInfo;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class FromStringDeserializer_ESTest extends FromStringDeserializer_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Class<CollectionLikeType> class0 = CollectionLikeType.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 3);
      // Undeclared exception!
      try { 
        fromStringDeserializer_Std0._deserialize("is&0j.}sA--f", (DeserializationContext) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Illegal character in path at index 6: is&0j.}sA--f
         //
         verifyException("java.net.URI", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Class<SimpleTimeZone> class0 = SimpleTimeZone.class;
      UUIDDeserializer uUIDDeserializer0 = new UUIDDeserializer();
      // Undeclared exception!
      try { 
        uUIDDeserializer0._deserializeEmbedded(class0, (DeserializationContext) null);
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
      Class<File> class0 = File.class;
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
      Class<CollectionLikeType> class0 = CollectionLikeType.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = FromStringDeserializer.findDeserializer(class0);
      assertNull(fromStringDeserializer_Std0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Class<URL> class0 = URL.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = FromStringDeserializer.findDeserializer(class0);
      assertEquals(5, FromStringDeserializer.Std.STD_JAVA_TYPE);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Class<URI> class0 = URI.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = FromStringDeserializer.findDeserializer(class0);
      URI uRI0 = (URI)fromStringDeserializer_Std0._deserializeFromEmptyString();
      assertEquals("", uRI0.getRawPath());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper((JsonFactory) null);
      ObjectReader objectReader0 = objectMapper0.reader();
      Class<SimpleType> class0 = SimpleType.class;
      ObjectReader objectReader1 = objectReader0.forType(class0);
      assertNotSame(objectReader1, objectReader0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Class<Currency> class0 = Currency.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = FromStringDeserializer.findDeserializer(class0);
      assertEquals(5, FromStringDeserializer.Std.STD_JAVA_TYPE);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Class<Pattern> class0 = Pattern.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = FromStringDeserializer.findDeserializer(class0);
      assertFalse(fromStringDeserializer_Std0.isCachable());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Class<Locale> class0 = Locale.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = FromStringDeserializer.findDeserializer(class0);
      assertEquals(6, FromStringDeserializer.Std.STD_CURRENCY);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Class<Charset> class0 = Charset.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = FromStringDeserializer.findDeserializer(class0);
      assertEquals(2, FromStringDeserializer.Std.STD_URL);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Class<TimeZone> class0 = TimeZone.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = FromStringDeserializer.findDeserializer(class0);
      assertEquals(1, FromStringDeserializer.Std.STD_FILE);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Class<InetAddress> class0 = InetAddress.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = FromStringDeserializer.findDeserializer(class0);
      assertEquals(6, FromStringDeserializer.Std.STD_CURRENCY);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Class<InetSocketAddress> class0 = InetSocketAddress.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = FromStringDeserializer.findDeserializer(class0);
      assertEquals(10, FromStringDeserializer.Std.STD_TIME_ZONE);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Class<StringBuilder> class0 = StringBuilder.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = FromStringDeserializer.findDeserializer(class0);
      Object object0 = fromStringDeserializer_Std0._deserializeFromEmptyString();
      assertEquals("", object0.toString());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      JsonFactory jsonFactory0 = new JsonFactory();
      JsonParser jsonParser0 = jsonFactory0.createParser("JSON");
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Class<Integer> class0 = Integer.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 2287);
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
  public void test17()  throws Throwable  {
      Class<CollectionLikeType> class0 = CollectionLikeType.class;
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 1);
      Object object0 = fromStringDeserializer_Std0._deserialize("[nl^u", defaultDeserializationContext_Impl0);
      assertEquals("[nl^u", object0.toString());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Class<CollectionLikeType> class0 = CollectionLikeType.class;
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 12);
      Object object0 = fromStringDeserializer_Std0._deserialize("[L_M(k]1l'7=$m=$$$", defaultDeserializationContext_Impl0);
      assertEquals("/200.42.42.0:0", object0.toString());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Class<CollectionLikeType> class0 = CollectionLikeType.class;
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 2);
      try { 
        fromStringDeserializer_Std0._deserialize("[=l", defaultDeserializationContext_Impl0);
        fail("Expecting exception: MalformedURLException");
      
      } catch(MalformedURLException e) {
         //
         // no protocol: [=l
         //
         verifyException("java.net.URL", e);
      }
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Class<CollectionLikeType> class0 = CollectionLikeType.class;
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 4);
      // Undeclared exception!
      try { 
        fromStringDeserializer_Std0._deserialize((String) null, defaultDeserializationContext_Impl0);
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
      Class<String> class0 = String.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 5);
      // Undeclared exception!
      try { 
        fromStringDeserializer_Std0._deserialize("A<aSRfu5=]-", (DeserializationContext) null);
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
      Class<CollectionLikeType> class0 = CollectionLikeType.class;
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 6);
      // Undeclared exception!
      try { 
        fromStringDeserializer_Std0._deserialize("Ma|F!:,HcWn", defaultDeserializationContext_Impl0);
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
      Class<CollectionLikeType> class0 = CollectionLikeType.class;
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 7);
      // Undeclared exception!
      try { 
        fromStringDeserializer_Std0._deserialize("[[B35#q>DvX 4Q$.a C", defaultDeserializationContext_Impl0);
        fail("Expecting exception: PatternSyntaxException");
      
      } catch(PatternSyntaxException e) {
         //
         // Unclosed character class near index 18
         // [[B35#q>DvX 4Q$.a C
         //                   ^
         //
         verifyException("java.util.regex.Pattern", e);
      }
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Class<Charset> class0 = Charset.class;
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 8);
      Object object0 = fromStringDeserializer_Std0._deserialize("[L_M(k]1l'7=$m=$$$", defaultDeserializationContext_Impl0);
      assertEquals("[l_M(K]1L'7=$M=$$$", object0.toString());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Class<CollectionLikeType> class0 = CollectionLikeType.class;
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 9);
      // Undeclared exception!
      try { 
        fromStringDeserializer_Std0._deserialize("Z5!~_oo}of+", defaultDeserializationContext_Impl0);
        fail("Expecting exception: IllegalCharsetNameException");
      
      } catch(IllegalCharsetNameException e) {
         //
         // Z5!~_oo}of+
         //
         verifyException("java.nio.charset.Charset", e);
      }
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Class<Locale> class0 = Locale.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 10);
      ZoneInfo zoneInfo0 = (ZoneInfo)fromStringDeserializer_Std0._deserialize("m", (DeserializationContext) null);
      assertEquals("GMT", zoneInfo0.getID());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Class<CollectionLikeType> class0 = CollectionLikeType.class;
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 11);
      Inet4Address inet4Address0 = (Inet4Address)fromStringDeserializer_Std0._deserialize("[L_(k]1l'7=m=$$$", defaultDeserializationContext_Impl0);
      assertFalse(inet4Address0.isMCOrgLocal());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Class<CollectionLikeType> class0 = CollectionLikeType.class;
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 13);
      Object object0 = fromStringDeserializer_Std0._deserialize("", defaultDeserializationContext_Impl0);
      assertEquals("", object0.toString());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Class<CollectionLikeType> class0 = CollectionLikeType.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 60);
      // Undeclared exception!
      try { 
        fromStringDeserializer_Std0._deserialize("is&0j.}sA--f", (DeserializationContext) null);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // Internal error: this code path should never get executed
         //
         verifyException("com.fasterxml.jackson.core.util.VersionUtil", e);
      }
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      Class<CollectionLikeType> class0 = CollectionLikeType.class;
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 8);
      Object object0 = fromStringDeserializer_Std0._deserialize("[=l", defaultDeserializationContext_Impl0);
      assertEquals("[=l", object0.toString());
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Class<TimeZone> class0 = TimeZone.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 8);
      Object object0 = fromStringDeserializer_Std0._deserialize("] -- unresolved forward-reference?", (DeserializationContext) null);
      assertEquals("] __ unresolved forward-reference?", object0.toString());
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      Class<CollectionLikeType> class0 = CollectionLikeType.class;
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 12);
      MockInetSocketAddress mockInetSocketAddress0 = (MockInetSocketAddress)fromStringDeserializer_Std0._deserialize("AnnotationIntrospector returned Converter definition of type ", defaultDeserializationContext_Impl0);
      assertEquals("200.42.42.0", mockInetSocketAddress0.getHostString());
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Class<CollectionLikeType> class0 = CollectionLikeType.class;
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 12);
      try { 
        fromStringDeserializer_Std0._deserialize("[Don't know how to convert embedded Object of type %s into %s", defaultDeserializationContext_Impl0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Bracketed IPv6 address must contain closing bracket
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.FromStringDeserializer$Std", e);
      }
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      Class<CollectionLikeType> class0 = CollectionLikeType.class;
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 12);
      // Undeclared exception!
      try { 
        fromStringDeserializer_Std0._deserialize("#,A`Q? A:VXG#", defaultDeserializationContext_Impl0);
        fail("Expecting exception: NumberFormatException");
      
      } catch(NumberFormatException e) {
         //
         // For input string: \"VXG#\"
         //
         verifyException("java.lang.NumberFormatException", e);
      }
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Class<Object> class0 = Object.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 12);
      MockInetSocketAddress mockInetSocketAddress0 = (MockInetSocketAddress)fromStringDeserializer_Std0._deserialize("+F:K3I-)+#X:", defaultDeserializationContext_Impl0);
      assertEquals(0, mockInetSocketAddress0.getPort());
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      Class<TimeZone> class0 = TimeZone.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 8);
      Locale locale0 = (Locale)fromStringDeserializer_Std0._deserializeFromEmptyString();
      assertEquals("", locale0.getCountry());
  }
}
