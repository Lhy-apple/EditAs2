/*
 * This file was automatically generated by EvoSuite
 * Sat Jul 29 19:01:04 GMT 2023
 */

package com.fasterxml.jackson.databind.deser.std;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.ObjectCodec;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JsonSerializable;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.deser.BeanDeserializerFactory;
import com.fasterxml.jackson.databind.deser.DefaultDeserializationContext;
import com.fasterxml.jackson.databind.deser.std.FromStringDeserializer;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.BooleanNode;
import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import com.fasterxml.jackson.databind.node.ValueNode;
import com.fasterxml.jackson.databind.type.ArrayType;
import com.fasterxml.jackson.databind.type.SimpleType;
import com.fasterxml.jackson.databind.util.RawValue;
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
import java.time.chrono.ChronoLocalDate;
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
      ObjectMapper objectMapper0 = new ObjectMapper();
      BooleanNode booleanNode0 = BooleanNode.TRUE;
      JsonParser jsonParser0 = objectMapper0.treeAsTokens(booleanNode0);
      Class<Currency> class0 = Currency.class;
      try { 
        objectMapper0.readValue(jsonParser0, class0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Can not construct instance of java.util.Currency from String value 'true': not a valid textual representation
         //  at [Source: java.lang.String@0000000002; line: -1, column: -1]
         //
         verifyException("com.fasterxml.jackson.databind.exc.InvalidFormatException", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      RawValue rawValue0 = new RawValue((JsonSerializable) null);
      ValueNode valueNode0 = jsonNodeFactory0.rawValueNode(rawValue0);
      JsonParser jsonParser0 = objectMapper0.treeAsTokens(valueNode0);
      Class<Currency> class0 = Currency.class;
      try { 
        objectMapper0.readValue(jsonParser0, class0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Don't know how to convert embedded Object of type com.fasterxml.jackson.databind.util.RawValue into java.util.Currency
         //  at [Source: java.lang.String@0000000002; line: -1, column: -1]
         //
         verifyException("com.fasterxml.jackson.databind.JsonMappingException", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Class<SimpleType> class0 = SimpleType.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 9);
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
      assertEquals(12, FromStringDeserializer.Std.STD_INET_SOCKET_ADDRESS);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Class<URL> class0 = URL.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = FromStringDeserializer.findDeserializer(class0);
      assertEquals(6, FromStringDeserializer.Std.STD_CURRENCY);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Class<URI> class0 = URI.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = FromStringDeserializer.findDeserializer(class0);
      assertEquals(2, FromStringDeserializer.Std.STD_URL);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Class<ArrayType> class0 = ArrayType.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = FromStringDeserializer.findDeserializer(class0);
      assertNull(fromStringDeserializer_Std0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Class<Pattern> class0 = Pattern.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = FromStringDeserializer.findDeserializer(class0);
      assertEquals(6, FromStringDeserializer.Std.STD_CURRENCY);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Class<Locale> class0 = Locale.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = FromStringDeserializer.findDeserializer(class0);
      assertFalse(fromStringDeserializer_Std0.isCachable());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Class<Charset> class0 = Charset.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = FromStringDeserializer.findDeserializer(class0);
      assertEquals(2, FromStringDeserializer.Std.STD_URL);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Class<TimeZone> class0 = TimeZone.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = FromStringDeserializer.findDeserializer(class0);
      assertEquals(7, FromStringDeserializer.Std.STD_PATTERN);
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
      assertEquals(9, FromStringDeserializer.Std.STD_CHARSET);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      ArrayNode arrayNode0 = objectMapper0.createArrayNode();
      JsonParser jsonParser0 = arrayNode0.traverse((ObjectCodec) objectMapper0);
      Class<Currency> class0 = Currency.class;
      try { 
        objectMapper0.readValue(jsonParser0, class0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Can not deserialize instance of java.util.Currency out of START_ARRAY token
         //  at [Source: java.lang.String@0000000002; line: -1, column: -1]
         //
         verifyException("com.fasterxml.jackson.databind.JsonMappingException", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ValueNode valueNode0 = jsonNodeFactory0.rawValueNode((RawValue) null);
      JsonParser jsonParser0 = objectMapper0.treeAsTokens(valueNode0);
      Class<Currency> class0 = Currency.class;
      Currency currency0 = objectMapper0.readValue(jsonParser0, class0);
      assertNull(currency0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Class<ArrayType> class0 = ArrayType.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 1);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      MockFile mockFile0 = (MockFile)fromStringDeserializer_Std0._deserialize("4m", defaultDeserializationContext_Impl0);
      assertEquals(0L, mockFile0.getUsableSpace());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Class<ChronoLocalDate> class0 = ChronoLocalDate.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 2);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      try { 
        fromStringDeserializer_Std0._deserialize("Mj>lUJar1NfPVLK", defaultDeserializationContext_Impl0);
        fail("Expecting exception: MalformedURLException");
      
      } catch(MalformedURLException e) {
         //
         // no protocol: Mj>lUJar1NfPVLK
         //
         verifyException("java.net.URL", e);
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Class<Integer> class0 = Integer.TYPE;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 3);
      URI uRI0 = (URI)fromStringDeserializer_Std0._deserialize("com.fasterxml.jackson.databind.deser.std.FromStringDeserializer$Std", (DeserializationContext) null);
      assertNull(uRI0.getRawAuthority());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Class<ArrayType> class0 = ArrayType.class;
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 4);
      try { 
        fromStringDeserializer_Std0._deserialize("1TF~[zn_c`|C&gv?$7", defaultDeserializationContext_Impl0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Can not construct instance of com.fasterxml.jackson.databind.type.ArrayType, problem: null
         //
         verifyException("com.fasterxml.jackson.databind.JsonMappingException", e);
      }
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Class<ArrayType> class0 = ArrayType.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 5);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      // Undeclared exception!
      try { 
        fromStringDeserializer_Std0._deserialize("~&)p", defaultDeserializationContext_Impl0);
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
      Class<ArrayType> class0 = ArrayType.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 7);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      // Undeclared exception!
      try { 
        fromStringDeserializer_Std0._deserialize("[", defaultDeserializationContext_Impl0);
        fail("Expecting exception: PatternSyntaxException");
      
      } catch(PatternSyntaxException e) {
         //
         // Unclosed character class near index 0
         // [
         // ^
         //
         verifyException("java.util.regex.Pattern", e);
      }
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Class<ArrayType> class0 = ArrayType.class;
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 8);
      Object object0 = fromStringDeserializer_Std0._deserialize("1TF~[zn_c`9C_gv?$7", defaultDeserializationContext_Impl0);
      assertEquals("1tf~[zn_C`9C_gv?$7", object0.toString());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Class<Currency> class0 = Currency.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 9);
      // Undeclared exception!
      try { 
        fromStringDeserializer_Std0._deserialize(" - ", (DeserializationContext) null);
        fail("Expecting exception: IllegalCharsetNameException");
      
      } catch(IllegalCharsetNameException e) {
         //
         //  - 
         //
         verifyException("java.nio.charset.Charset", e);
      }
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Class<ArrayType> class0 = ArrayType.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 10);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      ZoneInfo zoneInfo0 = (ZoneInfo)fromStringDeserializer_Std0._deserialize("p7'&ZwGNM", defaultDeserializationContext_Impl0);
      assertEquals("GMT", zoneInfo0.getID());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Class<ArrayType> class0 = ArrayType.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 11);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Inet4Address inet4Address0 = (Inet4Address)fromStringDeserializer_Std0._deserialize("[gsLaU@", defaultDeserializationContext_Impl0);
      assertFalse(inet4Address0.isLinkLocalAddress());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Class<ArrayType> class0 = ArrayType.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 12);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      MockInetSocketAddress mockInetSocketAddress0 = (MockInetSocketAddress)fromStringDeserializer_Std0._deserialize("[]gsLaU@", defaultDeserializationContext_Impl0);
      assertFalse(mockInetSocketAddress0.isUnresolved());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Class<Object> class0 = Object.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, (-4236));
      // Undeclared exception!
      try { 
        fromStringDeserializer_Std0._deserialize("E", (DeserializationContext) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.FromStringDeserializer$Std", e);
      }
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Class<ArrayType> class0 = ArrayType.class;
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 8);
      Object object0 = fromStringDeserializer_Std0._deserialize("1TF~[z;rc`|C gv?$7", defaultDeserializationContext_Impl0);
      assertEquals("1tf~[z;rc`|c gv?$7", object0.toString());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Class<ArrayType> class0 = ArrayType.class;
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 8);
      Object object0 = fromStringDeserializer_Std0._deserialize("1TF~[zn_c`|C&gv?$7", defaultDeserializationContext_Impl0);
      assertEquals("1tf~[zn_C`|C&GV?$7", object0.toString());
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      Class<ArrayType> class0 = ArrayType.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 12);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      // Undeclared exception!
      try { 
        fromStringDeserializer_Std0._deserialize("MVoL'4W-);:>R`f", defaultDeserializationContext_Impl0);
        fail("Expecting exception: NumberFormatException");
      
      } catch(NumberFormatException e) {
         //
         // For input string: \">R`f\"
         //
         verifyException("java.lang.NumberFormatException", e);
      }
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Class<ArrayType> class0 = ArrayType.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 12);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      try { 
        fromStringDeserializer_Std0._deserialize("[ >*z\"LTlgk42jYd~L:", defaultDeserializationContext_Impl0);
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
      Class<ArrayType> class0 = ArrayType.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 12);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      MockInetSocketAddress mockInetSocketAddress0 = (MockInetSocketAddress)fromStringDeserializer_Std0._deserialize("com.fasterxml.jackson.databind.deser.std.FromStringDeserializer$Std", defaultDeserializationContext_Impl0);
      assertEquals("200.42.42.0", mockInetSocketAddress0.getHostString());
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Class<ArrayType> class0 = ArrayType.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class0, 3);
      URI uRI0 = (URI)fromStringDeserializer_Std0._deserializeFromEmptyString();
      assertNull(uRI0.getRawFragment());
  }
}