/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 13:42:55 GMT 2023
 */

package com.fasterxml.jackson.databind;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.core.JsonLocation;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.filter.TokenFilter;
import com.fasterxml.jackson.core.filter.TokenFilterContext;
import com.fasterxml.jackson.core.util.JsonParserDelegate;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JsonMappingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.SerializerProvider;
import com.fasterxml.jackson.databind.deser.BeanDeserializerFactory;
import com.fasterxml.jackson.databind.deser.DefaultDeserializationContext;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.fasterxml.jackson.databind.ser.DefaultSerializerProvider;
import java.io.BufferedOutputStream;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.PipedOutputStream;
import java.sql.SQLDataException;
import java.sql.SQLInvalidAuthorizationSpecException;
import java.sql.SQLNonTransientConnectionException;
import java.sql.SQLSyntaxErrorException;
import java.sql.SQLTimeoutException;
import java.sql.SQLWarning;
import java.util.HashMap;
import java.util.List;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class JsonMappingException_ESTest extends JsonMappingException_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      JsonMappingException jsonMappingException0 = new JsonMappingException("overflow, value cannot be represented as 16-bit value");
      JsonMappingException.Reference jsonMappingException_Reference0 = new JsonMappingException.Reference(jsonMappingException0);
      jsonMappingException_Reference0.setIndex(1486);
      assertEquals(1486, jsonMappingException_Reference0.getIndex());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      SQLWarning sQLWarning0 = new SQLWarning();
      JsonMappingException.Reference jsonMappingException_Reference0 = new JsonMappingException.Reference(sQLWarning0, "overflow, value cannot be represented as 16-bitvalue");
      JsonMappingException.Reference jsonMappingException_Reference1 = (JsonMappingException.Reference)jsonMappingException_Reference0.writeReplace();
      assertEquals((-1), jsonMappingException_Reference1.getIndex());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      BufferedOutputStream bufferedOutputStream0 = new BufferedOutputStream(byteArrayOutputStream0, 2);
      JsonMappingException.Reference jsonMappingException_Reference0 = new JsonMappingException.Reference(bufferedOutputStream0, "JSON");
      Object object0 = jsonMappingException_Reference0.getFrom();
      assertEquals((-1), jsonMappingException_Reference0.getIndex());
      assertNotNull(object0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      SQLWarning sQLWarning0 = new SQLWarning("overflow, value cannot be represented as 16-bit value");
      JsonMappingException.Reference jsonMappingException_Reference0 = new JsonMappingException.Reference(sQLWarning0, "overflow, value cannot be represented as 16-bit value");
      jsonMappingException_Reference0.setFieldName("}st}<lp?e");
      assertEquals("}st}<lp?e", jsonMappingException_Reference0.getFieldName());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      SQLWarning sQLWarning0 = new SQLWarning("overflow, value cannot be represented as 16-bit value");
      JsonMappingException.Reference jsonMappingException_Reference0 = new JsonMappingException.Reference(sQLWarning0, "overflow, value cannot be represented as 16-bit value");
      jsonMappingException_Reference0.setDescription("overflow, value cannot be represented as 16-bit value");
      assertEquals((-1), jsonMappingException_Reference0.getIndex());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      JsonMappingException.Reference jsonMappingException_Reference0 = new JsonMappingException.Reference();
      jsonMappingException_Reference0.getFieldName();
      assertEquals((-1), jsonMappingException_Reference0.getIndex());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      SQLWarning sQLWarning0 = new SQLWarning();
      JsonMappingException.Reference jsonMappingException_Reference0 = new JsonMappingException.Reference(sQLWarning0, "overflow, value cannot be represented as 16-bit value");
      int int0 = jsonMappingException_Reference0.getIndex();
      assertEquals((-1), int0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      JsonLocation jsonLocation0 = JsonLocation.NA;
      JsonMappingException jsonMappingException0 = new JsonMappingException(";8V/Teabry5Gu<I%", jsonLocation0);
      jsonMappingException0.prependPath((Object) "overflow, value cannot be represented as 16-bit value", 500);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      SQLWarning sQLWarning0 = new SQLWarning("overflow, value cannot be represented as 16-bit value");
      TokenFilter tokenFilter0 = TokenFilter.INCLUDE_ALL;
      TokenFilterContext tokenFilterContext0 = TokenFilterContext.createRootContext(tokenFilter0);
      JsonLocation jsonLocation0 = tokenFilterContext0.getStartLocation(sQLWarning0);
      JsonMappingException jsonMappingException0 = new JsonMappingException("overflow, value cannot be represented as 16-bit value", jsonLocation0, sQLWarning0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      JsonMappingException jsonMappingException0 = JsonMappingException.from((JsonGenerator) null, "");
      assertNotNull(jsonMappingException0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      SQLWarning sQLWarning0 = new SQLWarning("overflow, value cannot be represented as 16-bit value");
      // Undeclared exception!
      try { 
        JsonMappingException.from((DeserializationContext) null, "5mGc2;lFf}.>L^57", (Throwable) sQLWarning0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.JsonMappingException", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      JsonMappingException jsonMappingException0 = JsonMappingException.from((DeserializationContext) defaultDeserializationContext_Impl0, "k~GS`jseAoh%sx(k");
      assertNotNull(jsonMappingException0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      SQLInvalidAuthorizationSpecException sQLInvalidAuthorizationSpecException0 = new SQLInvalidAuthorizationSpecException();
      JsonMappingException jsonMappingException0 = new JsonMappingException("$mVCN.vH 1Vnc", sQLInvalidAuthorizationSpecException0);
      Object object0 = jsonMappingException0.getProcessor();
      assertNull(object0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      SQLTimeoutException sQLTimeoutException0 = new SQLTimeoutException("", (Throwable) null);
      SQLSyntaxErrorException sQLSyntaxErrorException0 = new SQLSyntaxErrorException("", "", sQLTimeoutException0);
      SQLDataException sQLDataException0 = new SQLDataException("", sQLSyntaxErrorException0);
      JsonMappingException jsonMappingException0 = JsonMappingException.from((JsonGenerator) null, "UNKNOWN[3118]", (Throwable) sQLDataException0);
      assertNotNull(jsonMappingException0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      SQLWarning sQLWarning0 = new SQLWarning("overflow, value cannot be represented as 16-bit value");
      JsonFactory jsonFactory0 = new JsonFactory();
      JsonParser jsonParser0 = jsonFactory0.createNonBlockingByteArrayParser();
      JsonMappingException jsonMappingException0 = JsonMappingException.from(jsonParser0, "+E`f", (Throwable) sQLWarning0);
      assertNotNull(jsonMappingException0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      JsonMappingException jsonMappingException0 = JsonMappingException.from((SerializerProvider) defaultSerializerProvider_Impl0, "w5");
      assertNotNull(jsonMappingException0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      SQLWarning sQLWarning0 = new SQLWarning("overflow, value cannot be represented as 16-bit value");
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      JsonMappingException jsonMappingException0 = JsonMappingException.from((SerializerProvider) defaultSerializerProvider_Impl0, "com.fasterxml.jackson.core.Version", (Throwable) sQLWarning0);
      assertNotNull(jsonMappingException0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      SQLWarning sQLWarning0 = new SQLWarning();
      JsonMappingException.Reference jsonMappingException_Reference0 = new JsonMappingException.Reference();
      JsonMappingException jsonMappingException0 = JsonMappingException.wrapWithPath((Throwable) sQLWarning0, jsonMappingException_Reference0);
      Integer integer0 = new Integer(2295);
      // Undeclared exception!
      try { 
        jsonMappingException0.prependPath((Object) integer0, (String) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // Cannot pass null fieldName
         //
         verifyException("com.fasterxml.jackson.databind.JsonMappingException$Reference", e);
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      SQLWarning sQLWarning0 = new SQLWarning("overflow, value cannot be represented as 16-bit value", "overflow, value cannot be represented as 16-bit value");
      Object object0 = new Object();
      JsonMappingException jsonMappingException0 = JsonMappingException.wrapWithPath((Throwable) sQLWarning0, object0, "IGNORE_MERGE_FOR_UNMERGEABLE");
      assertNotNull(jsonMappingException0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      PipedOutputStream pipedOutputStream0 = new PipedOutputStream();
      byte[] byteArray0 = new byte[1];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      TokenFilter tokenFilter0 = TokenFilter.INCLUDE_ALL;
      TokenFilterContext tokenFilterContext0 = TokenFilterContext.createRootContext(tokenFilter0);
      JsonLocation jsonLocation0 = tokenFilterContext0.getStartLocation(pipedOutputStream0);
      JsonMappingException jsonMappingException0 = new JsonMappingException(byteArrayInputStream0, "", jsonLocation0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      HashMap<String, JsonNode> hashMap0 = new HashMap<String, JsonNode>();
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0, hashMap0);
      ArrayNode arrayNode0 = objectNode0.putArray("G}CI{Av(m1G/a<~EXT");
      JsonParser jsonParser0 = arrayNode0.traverse();
      JsonParserDelegate jsonParserDelegate0 = new JsonParserDelegate(jsonParser0);
      JsonMappingException jsonMappingException0 = JsonMappingException.from((JsonParser) jsonParserDelegate0, (String) null);
      assertNotNull(jsonMappingException0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      JsonMappingException jsonMappingException0 = new JsonMappingException("w");
      String string0 = jsonMappingException0.getLocalizedMessage();
      assertEquals("w", string0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      JsonMappingException jsonMappingException0 = new JsonMappingException("w");
      JsonMappingException jsonMappingException1 = JsonMappingException.fromUnexpectedIOE(jsonMappingException0);
      assertFalse(jsonMappingException1.equals((Object)jsonMappingException0));
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      SQLWarning sQLWarning0 = new SQLWarning("overflow, value cannot be represented as 16-bit value");
      JsonMappingException.Reference jsonMappingException_Reference0 = new JsonMappingException.Reference(sQLWarning0, "overflow, value cannot be represented as 16-bit value");
      JsonMappingException jsonMappingException0 = JsonMappingException.wrapWithPath((Throwable) sQLWarning0, jsonMappingException_Reference0);
      jsonMappingException0.getPathReference();
      String string0 = jsonMappingException0.toString();
      assertEquals((-1), jsonMappingException_Reference0.getIndex());
      assertEquals("com.fasterxml.jackson.databind.JsonMappingException: overflow, value cannot be represented as 16-bit value (through reference chain: java.sql.SQLWarning[\"overflow, value cannot be represented as 16-bit value\"])", string0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      SQLWarning sQLWarning0 = new SQLWarning();
      JsonMappingException.Reference jsonMappingException_Reference0 = new JsonMappingException.Reference();
      JsonMappingException jsonMappingException0 = JsonMappingException.wrapWithPath((Throwable) sQLWarning0, jsonMappingException_Reference0);
      Integer integer0 = new Integer(2295);
      JsonMappingException jsonMappingException1 = JsonMappingException.wrapWithPath((Throwable) jsonMappingException0, (Object) integer0, 2295);
      String string0 = jsonMappingException1._buildMessage();
      assertEquals("(was java.sql.SQLWarning) (through reference chain: java.lang.Integer[2295]->UNKNOWN[?])", string0);
      assertEquals((-1), jsonMappingException_Reference0.getIndex());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      SQLNonTransientConnectionException sQLNonTransientConnectionException0 = new SQLNonTransientConnectionException("", ".[s.#2%", 34, (Throwable) null);
      JsonMappingException jsonMappingException0 = JsonMappingException.wrapWithPath((Throwable) sQLNonTransientConnectionException0, (Object) "", 34);
      assertNotNull(jsonMappingException0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      JsonMappingException.Reference jsonMappingException_Reference0 = new JsonMappingException.Reference();
      SQLSyntaxErrorException sQLSyntaxErrorException0 = new SQLSyntaxErrorException("", "");
      JsonMappingException jsonMappingException0 = new JsonMappingException("", sQLSyntaxErrorException0);
      JsonMappingException.wrapWithPath((Throwable) jsonMappingException0, jsonMappingException_Reference0);
      assertEquals((-1), jsonMappingException_Reference0.getIndex());
      
      List<JsonMappingException.Reference> list0 = jsonMappingException0.getPath();
      assertTrue(list0.contains(jsonMappingException_Reference0));
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      JsonMappingException jsonMappingException0 = new JsonMappingException("overflow, value cannot be represented as 16-bit value");
      List<JsonMappingException.Reference> list0 = jsonMappingException0.getPath();
      assertEquals(0, list0.size());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      SQLSyntaxErrorException sQLSyntaxErrorException0 = new SQLSyntaxErrorException("", "");
      JsonMappingException jsonMappingException0 = new JsonMappingException("", sQLSyntaxErrorException0);
      StringBuilder stringBuilder0 = new StringBuilder((CharSequence) "->");
      jsonMappingException0._appendPathDesc(stringBuilder0);
      assertEquals("->", stringBuilder0.toString());
  }
}