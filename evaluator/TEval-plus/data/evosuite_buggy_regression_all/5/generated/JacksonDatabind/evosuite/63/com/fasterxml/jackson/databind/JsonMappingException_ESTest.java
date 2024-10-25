/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 23:08:03 GMT 2023
 */

package com.fasterxml.jackson.databind;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.core.JsonLocation;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.ObjectCodec;
import com.fasterxml.jackson.core.io.SerializedString;
import com.fasterxml.jackson.core.util.DefaultPrettyPrinter;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JsonMappingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializerProvider;
import com.fasterxml.jackson.databind.cfg.DeserializerFactoryConfig;
import com.fasterxml.jackson.databind.deser.BeanDeserializerFactory;
import com.fasterxml.jackson.databind.deser.DefaultDeserializationContext;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import com.fasterxml.jackson.databind.node.ValueNode;
import com.fasterxml.jackson.databind.ser.DefaultSerializerProvider;
import com.fasterxml.jackson.databind.util.RawValue;
import java.io.BufferedOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.PipedReader;
import java.io.Writer;
import java.sql.BatchUpdateException;
import java.sql.SQLFeatureNotSupportedException;
import java.sql.SQLTransactionRollbackException;
import java.sql.SQLTransientException;
import java.util.List;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.io.MockFile;
import org.evosuite.runtime.mock.java.io.MockPrintWriter;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class JsonMappingException_ESTest extends JsonMappingException_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      SQLFeatureNotSupportedException sQLFeatureNotSupportedException0 = new SQLFeatureNotSupportedException();
      JsonMappingException jsonMappingException0 = JsonMappingException.wrapWithPath((Throwable) sQLFeatureNotSupportedException0, (Object) "vBF`E3o2t*dOpF~-zJ.", "vBF`E3o2t*dOpF~-zJ.");
      JsonMappingException.from((JsonGenerator) null, "vBF`E3o2t*dOpF~-zJ.");
      String string0 = jsonMappingException0._buildMessage();
      assertEquals("(was java.sql.SQLFeatureNotSupportedException) (through reference chain: java.lang.String[\"vBF`E3o2t*dOpF~-zJ.\"])", string0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      JsonMappingException.Reference jsonMappingException_Reference0 = new JsonMappingException.Reference();
      jsonMappingException_Reference0.setIndex(3002);
      assertEquals(3002, jsonMappingException_Reference0.getIndex());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      PipedReader pipedReader0 = new PipedReader();
      JsonMappingException.Reference jsonMappingException_Reference0 = new JsonMappingException.Reference(pipedReader0);
      JsonMappingException.Reference jsonMappingException_Reference1 = (JsonMappingException.Reference)jsonMappingException_Reference0.writeReplace();
      assertEquals((-1), jsonMappingException_Reference1.getIndex());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      JsonMappingException.Reference jsonMappingException_Reference0 = new JsonMappingException.Reference();
      jsonMappingException_Reference0.getFrom();
      assertEquals((-1), jsonMappingException_Reference0.getIndex());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      PipedReader pipedReader0 = new PipedReader();
      JsonMappingException.Reference jsonMappingException_Reference0 = new JsonMappingException.Reference(pipedReader0);
      jsonMappingException_Reference0.setFieldName("");
      assertEquals((-1), jsonMappingException_Reference0.getIndex());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      SQLFeatureNotSupportedException sQLFeatureNotSupportedException0 = new SQLFeatureNotSupportedException();
      JsonMappingException jsonMappingException0 = JsonMappingException.wrapWithPath((Throwable) sQLFeatureNotSupportedException0, (Object) sQLFeatureNotSupportedException0, 3842);
      jsonMappingException0.prependPath((Object) sQLFeatureNotSupportedException0, 3842);
      assertEquals(0, sQLFeatureNotSupportedException0.getErrorCode());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      JsonMappingException.Reference jsonMappingException_Reference0 = new JsonMappingException.Reference();
      jsonMappingException_Reference0.setDescription("");
      assertEquals((-1), jsonMappingException_Reference0.getIndex());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      JsonMappingException.Reference jsonMappingException_Reference0 = new JsonMappingException.Reference();
      jsonMappingException_Reference0.getFieldName();
      assertEquals((-1), jsonMappingException_Reference0.getIndex());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream(3241);
      JsonMappingException.Reference jsonMappingException_Reference0 = new JsonMappingException.Reference(byteArrayOutputStream0, 0);
      int int0 = jsonMappingException_Reference0.getIndex();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      SQLFeatureNotSupportedException sQLFeatureNotSupportedException0 = new SQLFeatureNotSupportedException();
      JsonMappingException.Reference jsonMappingException_Reference0 = new JsonMappingException.Reference();
      JsonMappingException jsonMappingException0 = JsonMappingException.wrapWithPath((Throwable) sQLFeatureNotSupportedException0, jsonMappingException_Reference0);
      String string0 = jsonMappingException0.getPathReference();
      assertEquals((-1), jsonMappingException_Reference0.getIndex());
      assertEquals("UNKNOWN[?]", string0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      JsonMappingException jsonMappingException0 = new JsonMappingException("fzbv&6{9Ue", (JsonLocation) null, (Throwable) null);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      BatchUpdateException batchUpdateException0 = new BatchUpdateException();
      // Undeclared exception!
      try { 
        JsonMappingException.from((DeserializationContext) null, "eWiNc5&$W%", (Throwable) batchUpdateException0);
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
      // Undeclared exception!
      try { 
        JsonMappingException.from((DeserializationContext) null, "");
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.JsonMappingException", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      SQLFeatureNotSupportedException sQLFeatureNotSupportedException0 = new SQLFeatureNotSupportedException();
      JsonMappingException.Reference jsonMappingException_Reference0 = new JsonMappingException.Reference();
      JsonMappingException jsonMappingException0 = JsonMappingException.wrapWithPath((Throwable) sQLFeatureNotSupportedException0, jsonMappingException_Reference0);
      jsonMappingException0.getProcessor();
      assertEquals((-1), jsonMappingException_Reference0.getIndex());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      JsonMappingException jsonMappingException0 = new JsonMappingException("");
      List<JsonMappingException.Reference> list0 = jsonMappingException0.getPath();
      assertTrue(list0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      JsonFactory jsonFactory0 = new JsonFactory();
      File file0 = MockFile.createTempFile("JSON", "JSON");
      MockPrintWriter mockPrintWriter0 = new MockPrintWriter(file0);
      JsonGenerator jsonGenerator0 = jsonFactory0.createGenerator((Writer) mockPrintWriter0);
      SQLTransientException sQLTransientException0 = new SQLTransientException("java.io.PipedReader[?]");
      JsonMappingException jsonMappingException0 = JsonMappingException.from(jsonGenerator0, "JSON", (Throwable) sQLTransientException0);
      assertNotNull(jsonMappingException0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      SQLFeatureNotSupportedException sQLFeatureNotSupportedException0 = new SQLFeatureNotSupportedException();
      ObjectMapper objectMapper0 = new ObjectMapper((JsonFactory) null);
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory(false);
      SerializedString serializedString0 = DefaultPrettyPrinter.DEFAULT_ROOT_VALUE_SEPARATOR;
      RawValue rawValue0 = new RawValue(serializedString0);
      ValueNode valueNode0 = jsonNodeFactory0.rawValueNode(rawValue0);
      JsonParser jsonParser0 = objectMapper0.treeAsTokens(valueNode0);
      JsonMappingException.from(jsonParser0, "Finn", (Throwable) sQLFeatureNotSupportedException0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      JsonMappingException jsonMappingException0 = JsonMappingException.from((SerializerProvider) defaultSerializerProvider_Impl0, "");
      assertNotNull(jsonMappingException0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      SQLFeatureNotSupportedException sQLFeatureNotSupportedException0 = new SQLFeatureNotSupportedException();
      // Undeclared exception!
      try { 
        JsonMappingException.from((SerializerProvider) null, "8vH", (Throwable) sQLFeatureNotSupportedException0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.JsonMappingException", e);
      }
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      SQLFeatureNotSupportedException sQLFeatureNotSupportedException0 = new SQLFeatureNotSupportedException();
      JsonMappingException jsonMappingException0 = JsonMappingException.wrapWithPath((Throwable) sQLFeatureNotSupportedException0, (Object) sQLFeatureNotSupportedException0, 3868);
      String string0 = jsonMappingException0.toString();
      assertEquals("com.fasterxml.jackson.databind.JsonMappingException: (was java.sql.SQLFeatureNotSupportedException) (through reference chain: java.sql.SQLFeatureNotSupportedException[3868])", string0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      SQLFeatureNotSupportedException sQLFeatureNotSupportedException0 = new SQLFeatureNotSupportedException();
      JsonMappingException.Reference jsonMappingException_Reference0 = new JsonMappingException.Reference();
      JsonMappingException jsonMappingException0 = JsonMappingException.wrapWithPath((Throwable) sQLFeatureNotSupportedException0, jsonMappingException_Reference0);
      jsonMappingException0.prependPath((Object) sQLFeatureNotSupportedException0, "3[^/Q!.'9<WdF*");
      assertEquals((-1), jsonMappingException_Reference0.getIndex());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      JsonMappingException jsonMappingException0 = new JsonMappingException("V=mFxpF>", (Throwable) null);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      SQLFeatureNotSupportedException sQLFeatureNotSupportedException0 = new SQLFeatureNotSupportedException();
      JsonLocation jsonLocation0 = new JsonLocation(sQLFeatureNotSupportedException0, 2407L, (-4216), 628);
      JsonMappingException jsonMappingException0 = new JsonMappingException("JF", jsonLocation0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      PipedReader pipedReader0 = new PipedReader();
      JsonLocation jsonLocation0 = new JsonLocation(pipedReader0, 3451L, (-275), 411);
      JsonMappingException jsonMappingException0 = new JsonMappingException(pipedReader0, "com.fasterxml.jackson.databind.PropertyNamingStrategy$KebabCaseStrategy", jsonLocation0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Class<BufferedOutputStream> class0 = BufferedOutputStream.class;
      JsonMappingException jsonMappingException0 = defaultDeserializationContext_Impl0.instantiationException(class0, "");
      JsonMappingException.Reference jsonMappingException_Reference0 = new JsonMappingException.Reference();
      jsonMappingException0.prependPath(jsonMappingException_Reference0);
      JsonMappingException jsonMappingException1 = JsonMappingException.wrapWithPath((Throwable) jsonMappingException0, jsonMappingException_Reference0);
      jsonMappingException1._buildMessage();
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      SQLFeatureNotSupportedException sQLFeatureNotSupportedException0 = new SQLFeatureNotSupportedException();
      JsonMappingException.Reference jsonMappingException_Reference0 = new JsonMappingException.Reference();
      JsonMappingException jsonMappingException0 = JsonMappingException.wrapWithPath((Throwable) sQLFeatureNotSupportedException0, jsonMappingException_Reference0);
      String string0 = jsonMappingException0.getLocalizedMessage();
      assertEquals("(was java.sql.SQLFeatureNotSupportedException) (through reference chain: UNKNOWN[?])", string0);
      assertEquals((-1), jsonMappingException_Reference0.getIndex());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      SQLFeatureNotSupportedException sQLFeatureNotSupportedException0 = new SQLFeatureNotSupportedException();
      JsonMappingException.Reference jsonMappingException_Reference0 = new JsonMappingException.Reference();
      JsonMappingException jsonMappingException0 = JsonMappingException.wrapWithPath((Throwable) sQLFeatureNotSupportedException0, jsonMappingException_Reference0);
      JsonMappingException.fromUnexpectedIOE(jsonMappingException0);
      assertEquals((-1), jsonMappingException_Reference0.getIndex());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      JsonMappingException.Reference jsonMappingException_Reference0 = null;
      try {
        jsonMappingException_Reference0 = new JsonMappingException.Reference((Object) null, (String) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // Can not pass null fieldName
         //
         verifyException("com.fasterxml.jackson.databind.JsonMappingException$Reference", e);
      }
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Class<BufferedOutputStream> class0 = BufferedOutputStream.class;
      JsonMappingException jsonMappingException0 = defaultDeserializationContext_Impl0.instantiationException(class0, "");
      JsonMappingException.Reference jsonMappingException_Reference0 = new JsonMappingException.Reference(class0, 158);
      JsonMappingException.wrapWithPath((Throwable) jsonMappingException0, jsonMappingException_Reference0);
      String string0 = jsonMappingException0._buildMessage();
      assertEquals("Can not construct instance of java.io.BufferedOutputStream:  (through reference chain: java.io.BufferedOutputStream[158])", string0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      ArrayNode arrayNode0 = objectMapper0.createArrayNode();
      JsonParser jsonParser0 = arrayNode0.traverse((ObjectCodec) objectMapper0);
      JsonMappingException.from(jsonParser0, "y|5|\"5E<7fUDT&tv6");
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      SQLFeatureNotSupportedException sQLFeatureNotSupportedException0 = new SQLFeatureNotSupportedException(" ", " ");
      JsonMappingException.wrapWithPath((Throwable) sQLFeatureNotSupportedException0, (Object) sQLFeatureNotSupportedException0, 2);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      JsonMappingException.Reference jsonMappingException_Reference0 = new JsonMappingException.Reference();
      SQLTransactionRollbackException sQLTransactionRollbackException0 = new SQLTransactionRollbackException("");
      JsonMappingException.wrapWithPath((Throwable) sQLTransactionRollbackException0, jsonMappingException_Reference0);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      SQLFeatureNotSupportedException sQLFeatureNotSupportedException0 = new SQLFeatureNotSupportedException();
      JsonMappingException.Reference jsonMappingException_Reference0 = new JsonMappingException.Reference();
      JsonMappingException jsonMappingException0 = JsonMappingException.wrapWithPath((Throwable) sQLFeatureNotSupportedException0, jsonMappingException_Reference0);
      jsonMappingException0.getPath();
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory(deserializerFactoryConfig0);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Class<JsonMappingException> class0 = JsonMappingException.class;
      JsonMappingException jsonMappingException0 = defaultDeserializationContext_Impl0.weirdKeyException(class0, (String) null, "UNKNOWN[?]");
      JsonMappingException.fromUnexpectedIOE(jsonMappingException0);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      JsonMappingException jsonMappingException0 = new JsonMappingException("B");
      StringBuilder stringBuilder0 = new StringBuilder(2523);
      jsonMappingException0._appendPathDesc(stringBuilder0);
  }
}
