/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 03:09:08 GMT 2023
 */

package com.fasterxml.jackson.databind.deser.std;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.util.JsonParserSequence;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
import com.fasterxml.jackson.databind.deser.BeanDeserializerFactory;
import com.fasterxml.jackson.databind.deser.DefaultDeserializationContext;
import com.fasterxml.jackson.databind.deser.std.FromStringDeserializer;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import com.fasterxml.jackson.databind.node.NumericNode;
import com.fasterxml.jackson.databind.node.ValueNode;
import com.fasterxml.jackson.databind.type.CollectionLikeType;
import com.fasterxml.jackson.databind.util.RawValue;
import java.io.File;
import java.io.FilterInputStream;
import java.io.IOException;
import java.net.Inet4Address;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.net.MalformedURLException;
import java.net.URI;
import java.net.URL;
import java.nio.charset.Charset;
import java.util.Currency;
import java.util.Locale;
import java.util.TimeZone;
import java.util.regex.Pattern;
import java.util.regex.PatternSyntaxException;
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
      Class<Locale> class0 = Locale.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = FromStringDeserializer.findDeserializer(class0);
      assertEquals(6, FromStringDeserializer.Std.STD_CURRENCY);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      ObjectReader objectReader0 = objectMapper0.reader();
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.withExactBigDecimals(true);
      ArrayNode arrayNode0 = jsonNodeFactory0.arrayNode();
      RawValue rawValue0 = new RawValue("HXSa~Nk{Ge>fnC0");
      ValueNode valueNode0 = arrayNode0.rawValueNode(rawValue0);
      JsonParser jsonParser0 = objectReader0.treeAsTokens(valueNode0);
      JsonParserSequence jsonParserSequence0 = JsonParserSequence.createFlattened(jsonParser0, jsonParser0);
      jsonParserSequence0.nextToken();
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Class<Charset> class0 = Charset.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, (-1634));
      try { 
        fromStringDeserializer_Std0.deserialize((JsonParser) jsonParserSequence0, (DeserializationContext) defaultDeserializationContext_Impl0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Don't know how to convert embedded Object of type com.fasterxml.jackson.databind.util.RawValue into java.nio.charset.Charset
         //
         verifyException("com.fasterxml.jackson.databind.JsonMappingException", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Class<Object> class0 = Object.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 3692);
      Object object0 = fromStringDeserializer_Std0._deserializeFromEmptyString();
      assertNull(object0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Class<?>[] classArray0 = FromStringDeserializer.types();
      assertEquals(12, classArray0.length);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Class<File> class0 = File.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = FromStringDeserializer.findDeserializer(class0);
      assertEquals(1, FromStringDeserializer.Std.STD_FILE);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Class<URL> class0 = URL.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = FromStringDeserializer.findDeserializer(class0);
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ArrayNode arrayNode0 = jsonNodeFactory0.arrayNode();
      JsonParser jsonParser0 = arrayNode0.traverse();
      JsonParserSequence jsonParserSequence0 = JsonParserSequence.createFlattened(jsonParser0, jsonParser0);
      jsonParserSequence0.nextToken();
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      // Undeclared exception!
      try { 
        fromStringDeserializer_Std0.deserialize((JsonParser) jsonParserSequence0, (DeserializationContext) defaultDeserializationContext_Impl0);
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
      Class<URI> class0 = URI.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = FromStringDeserializer.findDeserializer(class0);
      assertEquals(11, FromStringDeserializer.Std.STD_INET_ADDRESS);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Class<Currency> class0 = Currency.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = FromStringDeserializer.findDeserializer(class0);
      assertFalse(fromStringDeserializer_Std0.isCachable());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Class<Pattern> class0 = Pattern.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = FromStringDeserializer.findDeserializer(class0);
      assertEquals(6, FromStringDeserializer.Std.STD_CURRENCY);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      Class<TimeZone> class0 = TimeZone.class;
      ObjectReader objectReader0 = objectMapper0.readerFor(class0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Class<Charset> class0 = Charset.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = FromStringDeserializer.findDeserializer(class0);
      assertEquals(12, FromStringDeserializer.Std.STD_INET_SOCKET_ADDRESS);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Class<Object> class0 = Object.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = FromStringDeserializer.findDeserializer(class0);
      assertNull(fromStringDeserializer_Std0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Class<InetAddress> class0 = InetAddress.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = FromStringDeserializer.findDeserializer(class0);
      assertEquals(9, FromStringDeserializer.Std.STD_CHARSET);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Class<InetSocketAddress> class0 = InetSocketAddress.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = FromStringDeserializer.findDeserializer(class0);
      assertEquals(10, FromStringDeserializer.Std.STD_TIME_ZONE);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory(false);
      NumericNode numericNode0 = jsonNodeFactory0.numberNode((byte)10);
      JsonParser jsonParser0 = objectMapper0.treeAsTokens(numericNode0);
      JsonParserSequence jsonParserSequence0 = JsonParserSequence.createFlattened(jsonParser0, jsonParser0);
      jsonParserSequence0.nextToken();
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Class<URI> class0 = URI.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, (byte)10);
      ZoneInfo zoneInfo0 = (ZoneInfo)fromStringDeserializer_Std0.deserialize((JsonParser) jsonParserSequence0, (DeserializationContext) defaultDeserializationContext_Impl0);
      assertEquals("GMT", zoneInfo0.getID());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      NumericNode numericNode0 = jsonNodeFactory0.numberNode((byte)54);
      JsonParser jsonParser0 = objectMapper0.treeAsTokens(numericNode0);
      JsonParserSequence jsonParserSequence0 = JsonParserSequence.createFlattened(jsonParser0, jsonParser0);
      jsonParserSequence0.nextToken();
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Class<URI> class0 = URI.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 9);
      // Undeclared exception!
      try { 
        fromStringDeserializer_Std0.deserialize((JsonParser) jsonParserSequence0, (DeserializationContext) defaultDeserializationContext_Impl0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.exc.InvalidFormatException", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      NumericNode numericNode0 = jsonNodeFactory0.numberNode((byte)54);
      JsonParser jsonParser0 = objectMapper0.treeAsTokens(numericNode0);
      JsonParserSequence jsonParserSequence0 = JsonParserSequence.createFlattened(jsonParser0, jsonParser0);
      jsonParserSequence0.nextToken();
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Class<URI> class0 = URI.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, (byte)54);
      // Undeclared exception!
      try { 
        fromStringDeserializer_Std0.deserialize((JsonParser) jsonParserSequence0, (DeserializationContext) defaultDeserializationContext_Impl0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.exc.InvalidFormatException", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      ObjectReader objectReader0 = objectMapper0.reader();
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.withExactBigDecimals(true);
      ArrayNode arrayNode0 = jsonNodeFactory0.arrayNode();
      RawValue rawValue0 = new RawValue("HXSa~Nk{Ge>fnC0");
      ValueNode valueNode0 = arrayNode0.rawValueNode(rawValue0);
      JsonParser jsonParser0 = objectReader0.treeAsTokens(valueNode0);
      JsonParserSequence jsonParserSequence0 = JsonParserSequence.createFlattened(jsonParser0, jsonParser0);
      jsonParserSequence0.nextToken();
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Class<Object> class0 = Object.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 3811);
      Object object0 = fromStringDeserializer_Std0.deserialize(jsonParser0, (DeserializationContext) defaultDeserializationContext_Impl0);
      assertSame(rawValue0, object0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Class<Locale> class0 = Locale.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 1);
      MockFile mockFile0 = (MockFile)fromStringDeserializer_Std0._deserialize("Lx2", (DeserializationContext) null);
      assertNull(mockFile0.getParent());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Class<Currency> class0 = Currency.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 2);
      try { 
        fromStringDeserializer_Std0._deserialize("[Dn=}:LYEHb", (DeserializationContext) null);
        fail("Expecting exception: MalformedURLException");
      
      } catch(MalformedURLException e) {
         //
         // no protocol: [Dn=}:LYEHb
         //
         verifyException("java.net.URL", e);
      }
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Class<TimeZone> class0 = TimeZone.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 3);
      // Undeclared exception!
      try { 
        fromStringDeserializer_Std0._deserialize("[' value but there was more than a single value in the array", (DeserializationContext) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Illegal character in path at index 0: [' value but there was more than a single value in the array
         //
         verifyException("java.net.URI", e);
      }
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Class<URL> class0 = URL.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 4);
      // Undeclared exception!
      try { 
        fromStringDeserializer_Std0._deserialize(")", (DeserializationContext) null);
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
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 5);
      // Undeclared exception!
      try { 
        fromStringDeserializer_Std0._deserialize((String) null, (DeserializationContext) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.FromStringDeserializer$Std", e);
      }
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Class<Locale> class0 = Locale.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 6);
      // Undeclared exception!
      try { 
        fromStringDeserializer_Std0._deserialize("[zb9_!gWgw]ih!\"", (DeserializationContext) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("java.util.Currency", e);
      }
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Class<Object> class0 = Object.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 7);
      // Undeclared exception!
      try { 
        fromStringDeserializer_Std0._deserialize("[&?bv9m'&:DdI", (DeserializationContext) null);
        fail("Expecting exception: PatternSyntaxException");
      
      } catch(PatternSyntaxException e) {
         //
         // Unclosed character class near index 12
         // [&?bv9m'&:DdI
         //             ^
         //
         verifyException("java.util.regex.Pattern", e);
      }
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Class<Object> class0 = Object.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 11);
      Inet4Address inet4Address0 = (Inet4Address)fromStringDeserializer_Std0._deserialize("w|=9j>6[F", (DeserializationContext) null);
      assertFalse(inet4Address0.isMCGlobal());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Class<FilterInputStream> class0 = FilterInputStream.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 12);
      MockInetSocketAddress mockInetSocketAddress0 = (MockInetSocketAddress)fromStringDeserializer_Std0._deserialize("Problem binding JSON into Map.Entry: unexpected content after JSON Object entry: ", (DeserializationContext) null);
      assertFalse(mockInetSocketAddress0.isUnresolved());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Class<Locale> class0 = Locale.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 8);
      Object object0 = fromStringDeserializer_Std0._deserialize("[zb9_!gWgw]ih!\"", (DeserializationContext) null);
      assertEquals("[zb9_!GWGW]IH!\"", object0.toString());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Class<URL> class0 = URL.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 8);
      Locale locale0 = (Locale)fromStringDeserializer_Std0._deserialize("<P$tC", (DeserializationContext) null);
      assertEquals("<p$tc", locale0.getLanguage());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Class<URI> class0 = URI.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 8);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Object object0 = fromStringDeserializer_Std0._deserialize("P_^xai(X_j2}", defaultDeserializationContext_Impl0);
      assertEquals("p_^XAI(X_j2}", object0.toString());
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      Class<FilterInputStream> class0 = FilterInputStream.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 12);
      MockInetSocketAddress mockInetSocketAddress0 = (MockInetSocketAddress)fromStringDeserializer_Std0._deserialize("[zb9_!gWgw]ih!\"", (DeserializationContext) null);
      assertEquals("200.42.42.0", mockInetSocketAddress0.getHostString());
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Class<Object> class0 = Object.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 12);
      try { 
        fromStringDeserializer_Std0._deserialize("[&?bv9m'&:DdI", (DeserializationContext) null);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Bracketed IPv6 address must contain closing bracket
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.FromStringDeserializer$Std", e);
      }
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      Class<Charset> class0 = Charset.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 12);
      Object object0 = fromStringDeserializer_Std0._deserialize("com.fasterxml.jackson.databind.deser.std.FromStringDeserializer", (DeserializationContext) null);
      assertEquals("/200.42.42.0:0", object0.toString());
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Class<URI> class0 = URI.class;
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 12);
      // Undeclared exception!
      try { 
        fromStringDeserializer_Std0._deserialize("wmO!ihj.]1:>swPkMqB", defaultDeserializationContext_Impl0);
        fail("Expecting exception: NumberFormatException");
      
      } catch(NumberFormatException e) {
         //
         // For input string: \">swPkMqB\"
         //
         verifyException("java.lang.NumberFormatException", e);
      }
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      Class<InetSocketAddress> class0 = InetSocketAddress.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 3);
      URI uRI0 = (URI)fromStringDeserializer_Std0._deserializeFromEmptyString();
      assertFalse(uRI0.isOpaque());
  }
}
