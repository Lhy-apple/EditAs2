/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 21:45:10 GMT 2023
 */

package com.fasterxml.jackson.databind;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.core.JsonLocation;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.filter.FilteringParserDelegate;
import com.fasterxml.jackson.core.util.JsonParserSequence;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JsonMappingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
import com.fasterxml.jackson.databind.SerializerProvider;
import com.fasterxml.jackson.databind.cfg.DeserializerFactoryConfig;
import com.fasterxml.jackson.databind.deser.BeanDeserializerFactory;
import com.fasterxml.jackson.databind.deser.DefaultDeserializationContext;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.BooleanNode;
import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import com.fasterxml.jackson.databind.ser.DefaultSerializerProvider;
import java.io.CharArrayWriter;
import java.sql.SQLDataException;
import java.sql.SQLFeatureNotSupportedException;
import java.util.List;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.io.MockIOException;
import org.evosuite.runtime.mock.java.lang.MockThrowable;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class JsonMappingException_ESTest extends JsonMappingException_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      JsonMappingException.Reference jsonMappingException_Reference0 = new JsonMappingException.Reference();
      jsonMappingException_Reference0.setIndex(1002);
      assertEquals(1002, jsonMappingException_Reference0.getIndex());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      JsonMappingException.Reference jsonMappingException_Reference0 = new JsonMappingException.Reference();
      JsonMappingException.Reference jsonMappingException_Reference1 = (JsonMappingException.Reference)jsonMappingException_Reference0.writeReplace();
      assertEquals((-1), jsonMappingException_Reference1.getIndex());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      JsonMappingException.Reference jsonMappingException_Reference0 = new JsonMappingException.Reference((Object) null, (-924));
      jsonMappingException_Reference0.getFrom();
      assertEquals((-924), jsonMappingException_Reference0.getIndex());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      MockThrowable mockThrowable0 = new MockThrowable("");
      JsonMappingException.Reference jsonMappingException_Reference0 = new JsonMappingException.Reference(mockThrowable0, "");
      jsonMappingException_Reference0.setFieldName("");
      assertEquals((-1), jsonMappingException_Reference0.getIndex());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      SQLFeatureNotSupportedException sQLFeatureNotSupportedException0 = new SQLFeatureNotSupportedException();
      JsonMappingException.Reference jsonMappingException_Reference0 = new JsonMappingException.Reference(sQLFeatureNotSupportedException0);
      JsonMappingException jsonMappingException0 = JsonMappingException.wrapWithPath((Throwable) sQLFeatureNotSupportedException0, jsonMappingException_Reference0);
      jsonMappingException0.prependPath((Object) jsonMappingException_Reference0, 389);
      assertEquals((-1), jsonMappingException_Reference0.getIndex());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      MockThrowable mockThrowable0 = new MockThrowable("");
      JsonMappingException.Reference jsonMappingException_Reference0 = new JsonMappingException.Reference(mockThrowable0, "");
      jsonMappingException_Reference0.setDescription("");
      String string0 = jsonMappingException_Reference0.getDescription();
      assertEquals("", string0);
      assertEquals((-1), jsonMappingException_Reference0.getIndex());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      JsonMappingException.Reference jsonMappingException_Reference0 = new JsonMappingException.Reference((Object) null, 500);
      jsonMappingException_Reference0.getFieldName();
      assertEquals(500, jsonMappingException_Reference0.getIndex());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      MockThrowable mockThrowable0 = new MockThrowable("");
      JsonMappingException.Reference jsonMappingException_Reference0 = new JsonMappingException.Reference(mockThrowable0, "");
      int int0 = jsonMappingException_Reference0.getIndex();
      assertEquals((-1), int0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      MockIOException mockIOException0 = new MockIOException();
      JsonLocation jsonLocation0 = JsonLocation.NA;
      JsonMappingException jsonMappingException0 = new JsonMappingException("6!Okd R^#", jsonLocation0, mockIOException0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      JsonMappingException jsonMappingException0 = JsonMappingException.from((JsonGenerator) null, "");
      JsonMappingException jsonMappingException1 = JsonMappingException.from((JsonGenerator) null, "", (Throwable) jsonMappingException0);
      assertFalse(jsonMappingException1.equals((Object)jsonMappingException0));
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      SQLDataException sQLDataException0 = new SQLDataException("CnnTb5z#k#PHqlj0", "CnnTb5z#k#PHqlj0", (-1695));
      // Undeclared exception!
      try { 
        JsonMappingException.from((DeserializationContext) null, "CnnTb5z#k#PHqlj0", (Throwable) sQLDataException0);
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
      // Undeclared exception!
      try { 
        JsonMappingException.from((DeserializationContext) null, (String) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.JsonMappingException", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      JsonMappingException jsonMappingException0 = new JsonMappingException("L|<", (Throwable) null);
      Object object0 = jsonMappingException0.getProcessor();
      assertNull(object0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      SQLFeatureNotSupportedException sQLFeatureNotSupportedException0 = new SQLFeatureNotSupportedException();
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory(deserializerFactoryConfig0);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Class<FilteringParserDelegate> class0 = FilteringParserDelegate.class;
      JsonMappingException jsonMappingException0 = defaultDeserializationContext_Impl0.instantiationException((Class<?>) class0, (Throwable) sQLFeatureNotSupportedException0);
      JsonMappingException.Reference jsonMappingException_Reference0 = new JsonMappingException.Reference(class0, 46);
      jsonMappingException0.prependPath(jsonMappingException_Reference0);
      String string0 = jsonMappingException0._buildMessage();
      assertEquals("Can not construct instance of com.fasterxml.jackson.core.filter.FilteringParserDelegate, problem: null (through reference chain: com.fasterxml.jackson.core.filter.FilteringParserDelegate[46])", string0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      // Undeclared exception!
      try { 
        JsonMappingException.wrapWithPath((Throwable) null, (Object) null, 1777);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      // Undeclared exception!
      try { 
        JsonMappingException.from((SerializerProvider) null, "xJ=WoUIP y|Ef");
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.JsonMappingException", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      SQLDataException sQLDataException0 = new SQLDataException();
      JsonMappingException jsonMappingException0 = JsonMappingException.from((SerializerProvider) defaultSerializerProvider_Impl0, "", (Throwable) sQLDataException0);
      assertNotNull(jsonMappingException0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      JsonMappingException jsonMappingException0 = JsonMappingException.from((JsonGenerator) null, "");
      String string0 = jsonMappingException0.toString();
      assertEquals("com.fasterxml.jackson.databind.JsonMappingException: ", string0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      SQLFeatureNotSupportedException sQLFeatureNotSupportedException0 = new SQLFeatureNotSupportedException();
      JsonMappingException.Reference jsonMappingException_Reference0 = new JsonMappingException.Reference();
      JsonMappingException jsonMappingException0 = JsonMappingException.wrapWithPath((Throwable) sQLFeatureNotSupportedException0, jsonMappingException_Reference0);
      jsonMappingException0.prependPath((Object) jsonMappingException_Reference0, "");
      String string0 = jsonMappingException0._buildMessage();
      assertEquals((-1), jsonMappingException_Reference0.getIndex());
      assertEquals("(was java.sql.SQLFeatureNotSupportedException) (through reference chain: com.fasterxml.jackson.databind.Reference[\"\"]->UNKNOWN[?])", string0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      JsonMappingException jsonMappingException0 = new JsonMappingException("Multiple 'any-setters' defined (", (JsonLocation) null);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      CharArrayWriter charArrayWriter0 = new CharArrayWriter();
      JsonMappingException jsonMappingException0 = new JsonMappingException(charArrayWriter0, "Multiple 'any-setters' defined (", (JsonLocation) null);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      ArrayNode arrayNode0 = new ArrayNode((JsonNodeFactory) null);
      JsonParser jsonParser0 = arrayNode0.traverse();
      JsonParserSequence jsonParserSequence0 = JsonParserSequence.createFlattened(false, jsonParser0, jsonParser0);
      JsonMappingException jsonMappingException0 = JsonMappingException.from((JsonParser) jsonParserSequence0, "ENtd+Fe4XEAhm");
      assertNotNull(jsonMappingException0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      SQLFeatureNotSupportedException sQLFeatureNotSupportedException0 = new SQLFeatureNotSupportedException();
      JsonMappingException.Reference jsonMappingException_Reference0 = new JsonMappingException.Reference(sQLFeatureNotSupportedException0);
      JsonMappingException jsonMappingException0 = JsonMappingException.wrapWithPath((Throwable) sQLFeatureNotSupportedException0, jsonMappingException_Reference0);
      String string0 = jsonMappingException0.getLocalizedMessage();
      assertEquals("(was java.sql.SQLFeatureNotSupportedException) (through reference chain: java.sql.SQLFeatureNotSupportedException[?])", string0);
      assertEquals((-1), jsonMappingException_Reference0.getIndex());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      MockIOException mockIOException0 = new MockIOException();
      JsonMappingException jsonMappingException0 = JsonMappingException.fromUnexpectedIOE(mockIOException0);
      assertNotNull(jsonMappingException0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      CharArrayWriter charArrayWriter0 = new CharArrayWriter();
      JsonMappingException.Reference jsonMappingException_Reference0 = null;
      try {
        jsonMappingException_Reference0 = new JsonMappingException.Reference(charArrayWriter0, (String) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // Can not pass null fieldName
         //
         verifyException("com.fasterxml.jackson.databind.JsonMappingException$Reference", e);
      }
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      SQLFeatureNotSupportedException sQLFeatureNotSupportedException0 = new SQLFeatureNotSupportedException();
      JsonMappingException.Reference jsonMappingException_Reference0 = new JsonMappingException.Reference(sQLFeatureNotSupportedException0);
      JsonMappingException jsonMappingException0 = JsonMappingException.wrapWithPath((Throwable) sQLFeatureNotSupportedException0, jsonMappingException_Reference0);
      JsonFactory jsonFactory0 = new JsonFactory();
      ObjectMapper objectMapper0 = new ObjectMapper(jsonFactory0);
      ObjectReader objectReader0 = objectMapper0.readerForUpdating(jsonMappingException_Reference0);
      BooleanNode booleanNode0 = BooleanNode.FALSE;
      JsonParser jsonParser0 = objectReader0.treeAsTokens(booleanNode0);
      JsonMappingException jsonMappingException1 = new JsonMappingException(jsonParser0, "JSON", jsonMappingException0);
      assertFalse(jsonMappingException1.equals((Object)jsonMappingException0));
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      JsonMappingException.Reference jsonMappingException_Reference0 = new JsonMappingException.Reference("");
      SQLFeatureNotSupportedException sQLFeatureNotSupportedException0 = new SQLFeatureNotSupportedException("");
      JsonMappingException.wrapWithPath((Throwable) sQLFeatureNotSupportedException0, jsonMappingException_Reference0);
      assertEquals((-1), jsonMappingException_Reference0.getIndex());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      MockIOException mockIOException0 = new MockIOException("nXP#6");
      JsonMappingException jsonMappingException0 = JsonMappingException.wrapWithPath((Throwable) mockIOException0, (JsonMappingException.Reference) null);
      assertNotNull(jsonMappingException0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      SQLFeatureNotSupportedException sQLFeatureNotSupportedException0 = new SQLFeatureNotSupportedException();
      JsonMappingException.Reference jsonMappingException_Reference0 = new JsonMappingException.Reference(sQLFeatureNotSupportedException0);
      JsonMappingException jsonMappingException0 = JsonMappingException.wrapWithPath((Throwable) sQLFeatureNotSupportedException0, jsonMappingException_Reference0);
      List<JsonMappingException.Reference> list0 = jsonMappingException0.getPath();
      assertTrue(list0.contains(jsonMappingException_Reference0));
      assertEquals((-1), jsonMappingException_Reference0.getIndex());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      JsonMappingException jsonMappingException0 = JsonMappingException.from((JsonGenerator) null, "");
      List<JsonMappingException.Reference> list0 = jsonMappingException0.getPath();
      assertEquals(0, list0.size());
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      JsonFactory jsonFactory0 = new JsonFactory(objectMapper0);
      objectMapper0.writeValueAsString(jsonFactory0);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      JsonMappingException jsonMappingException0 = new JsonMappingException("9S^_/Q7= Z^klv{A");
      String string0 = jsonMappingException0.getPathReference();
      assertEquals("", string0);
  }
}