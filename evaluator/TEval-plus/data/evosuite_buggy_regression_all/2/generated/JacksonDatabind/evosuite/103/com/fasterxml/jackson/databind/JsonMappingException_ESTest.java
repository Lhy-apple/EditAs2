/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 15:05:40 GMT 2023
 */

package com.fasterxml.jackson.databind;

import org.junit.Test;
import static org.junit.Assert.*;
import com.fasterxml.jackson.core.JsonEncoding;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.core.JsonLocation;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.ObjectCodec;
import com.fasterxml.jackson.core.util.JsonGeneratorDelegate;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JsonMappingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializerProvider;
import com.fasterxml.jackson.databind.cfg.DeserializerFactoryConfig;
import com.fasterxml.jackson.databind.deser.BeanDeserializerFactory;
import com.fasterxml.jackson.databind.deser.DefaultDeserializationContext;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.fasterxml.jackson.databind.ser.DefaultSerializerProvider;
import java.io.BufferedInputStream;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataOutput;
import java.io.LineNumberReader;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.sql.SQLIntegrityConstraintViolationException;
import java.sql.SQLInvalidAuthorizationSpecException;
import java.sql.SQLNonTransientConnectionException;
import java.sql.SQLNonTransientException;
import java.sql.SQLRecoverableException;
import java.sql.SQLTransientConnectionException;
import java.sql.SQLTransientException;
import java.sql.SQLWarning;
import java.util.HashMap;
import java.util.List;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.io.MockPrintStream;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class JsonMappingException_ESTest extends JsonMappingException_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      JsonMappingException.Reference jsonMappingException_Reference0 = new JsonMappingException.Reference();
      assertEquals((-1), jsonMappingException_Reference0.getIndex());
      
      jsonMappingException_Reference0.setIndex(0);
      SQLNonTransientException sQLNonTransientException0 = new SQLNonTransientException("", "");
      JsonMappingException jsonMappingException0 = JsonMappingException.wrapWithPath((Throwable) sQLNonTransientException0, jsonMappingException_Reference0);
      JsonMappingException.fromUnexpectedIOE(jsonMappingException0);
      assertEquals(0, jsonMappingException_Reference0.getIndex());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      JsonMappingException jsonMappingException0 = new JsonMappingException("0|\"8\u0001h3<t#\"<[;eDxR");
      JsonMappingException.Reference jsonMappingException_Reference0 = new JsonMappingException.Reference(jsonMappingException0, ".r;%");
      JsonMappingException.Reference jsonMappingException_Reference1 = (JsonMappingException.Reference)jsonMappingException_Reference0.writeReplace();
      assertEquals((-1), jsonMappingException_Reference1.getIndex());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      JsonMappingException.Reference jsonMappingException_Reference0 = new JsonMappingException.Reference();
      jsonMappingException_Reference0.getFrom();
      assertEquals((-1), jsonMappingException_Reference0.getIndex());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      JsonMappingException.Reference jsonMappingException_Reference0 = new JsonMappingException.Reference("-Infinity", (-227));
      jsonMappingException_Reference0.setFieldName("ff)M{");
      assertEquals((-227), jsonMappingException_Reference0.getIndex());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Class<LineNumberReader> class0 = LineNumberReader.class;
      JsonMappingException.Reference jsonMappingException_Reference0 = new JsonMappingException.Reference(class0);
      jsonMappingException_Reference0.getFieldName();
      assertEquals((-1), jsonMappingException_Reference0.getIndex());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      JsonMappingException.Reference jsonMappingException_Reference0 = new JsonMappingException.Reference();
      jsonMappingException_Reference0.setDescription("(6T}5RE )S]a1|");
      assertEquals((-1), jsonMappingException_Reference0.getIndex());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      JsonMappingException jsonMappingException0 = new JsonMappingException("pXA-^?02x");
      JsonMappingException.Reference jsonMappingException_Reference0 = new JsonMappingException.Reference(jsonMappingException0);
      int int0 = jsonMappingException_Reference0.getIndex();
      assertEquals((-1), int0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Class<LineNumberReader> class0 = LineNumberReader.class;
      JsonMappingException jsonMappingException0 = defaultDeserializationContext_Impl0.weirdKeyException(class0, (String) null, "IP_ADDRESS");
      jsonMappingException0.prependPath((Object) defaultDeserializationContext_Impl0, 2060);
      assertEquals(0, defaultDeserializationContext_Impl0.getDeserializationFeatures());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      JsonLocation jsonLocation0 = new JsonLocation("f6", (-212L), 1921, 1921);
      SQLIntegrityConstraintViolationException sQLIntegrityConstraintViolationException0 = new SQLIntegrityConstraintViolationException("1MDX*BhycL6l", "1MDX*BhycL6l", 500);
      SQLWarning sQLWarning0 = new SQLWarning(sQLIntegrityConstraintViolationException0);
      JsonMappingException jsonMappingException0 = new JsonMappingException("f6", jsonLocation0, sQLWarning0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      JsonFactory jsonFactory0 = new JsonFactory((ObjectCodec) null);
      MockPrintStream mockPrintStream0 = new MockPrintStream("Z7q)1l`ku/oBM");
      JsonEncoding jsonEncoding0 = JsonEncoding.UTF16_BE;
      JsonGenerator jsonGenerator0 = jsonFactory0.createGenerator((OutputStream) mockPrintStream0, jsonEncoding0);
      JsonGeneratorDelegate jsonGeneratorDelegate0 = new JsonGeneratorDelegate(jsonGenerator0, true);
      JsonMappingException jsonMappingException0 = JsonMappingException.from((JsonGenerator) jsonGeneratorDelegate0, "com.fasterxml.jackson.databind.deser.AbstractDeserializer");
      String string0 = jsonMappingException0.toString();
      assertEquals("com.fasterxml.jackson.databind.JsonMappingException: com.fasterxml.jackson.databind.deser.AbstractDeserializer", string0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory(deserializerFactoryConfig0);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      JsonMappingException jsonMappingException0 = JsonMappingException.from((DeserializationContext) defaultDeserializationContext_Impl0, "com.fasterxml.jackson.databind.PropertyMetadata$MergeInfo");
      assertNotNull(jsonMappingException0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      SQLRecoverableException sQLRecoverableException0 = new SQLRecoverableException("u[+~H", "(5Y{");
      SQLTransientConnectionException sQLTransientConnectionException0 = new SQLTransientConnectionException();
      JsonMappingException jsonMappingException0 = JsonMappingException.wrapWithPath((Throwable) sQLRecoverableException0, (Object) sQLTransientConnectionException0, (-658));
      Object object0 = jsonMappingException0.getProcessor();
      assertNull(object0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      SQLNonTransientException sQLNonTransientException0 = new SQLNonTransientException("", "");
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      JsonFactory jsonFactory0 = new JsonFactory();
      ObjectOutputStream objectOutputStream0 = new ObjectOutputStream(byteArrayOutputStream0);
      JsonGenerator jsonGenerator0 = jsonFactory0.createGenerator((DataOutput) objectOutputStream0);
      JsonMappingException jsonMappingException0 = JsonMappingException.from(jsonGenerator0, "PDmx5s4", (Throwable) sQLNonTransientException0);
      assertNotNull(jsonMappingException0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      JsonMappingException jsonMappingException0 = JsonMappingException.from((SerializerProvider) defaultSerializerProvider_Impl0, "K@COyJvBOo{f>f]");
      assertNotNull(jsonMappingException0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      SerializerProvider serializerProvider0 = objectMapper0.getSerializerProviderInstance();
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Class<Throwable> class0 = Throwable.class;
      JsonMappingException jsonMappingException0 = defaultDeserializationContext_Impl0.weirdKeyException(class0, "&4g6d[h", (String) null);
      JsonMappingException jsonMappingException1 = JsonMappingException.from(serializerProvider0, "r", (Throwable) jsonMappingException0);
      assertNotNull(jsonMappingException1);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      SQLRecoverableException sQLRecoverableException0 = new SQLRecoverableException("u[+~H", "(5Y{");
      SQLTransientConnectionException sQLTransientConnectionException0 = new SQLTransientConnectionException();
      JsonMappingException jsonMappingException0 = JsonMappingException.wrapWithPath((Throwable) sQLRecoverableException0, (Object) sQLTransientConnectionException0, (-658));
      SQLInvalidAuthorizationSpecException sQLInvalidAuthorizationSpecException0 = new SQLInvalidAuthorizationSpecException("&t0:FS@", "&t0:FS@", jsonMappingException0);
      JsonMappingException.wrapWithPath((Throwable) jsonMappingException0, (Object) sQLInvalidAuthorizationSpecException0, "RP%,VOY=A");
      String string0 = jsonMappingException0.getMessage();
      assertEquals("u[+~H (through reference chain: java.sql.SQLInvalidAuthorizationSpecException[\"RP%,VOY=A\"]->java.sql.SQLTransientConnectionException[?])", string0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      JsonMappingException jsonMappingException0 = new JsonMappingException("WmJm{}~1~C", (JsonLocation) null);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory(false);
      HashMap<String, JsonNode> hashMap0 = new HashMap<String, JsonNode>();
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0, hashMap0);
      ArrayNode arrayNode0 = objectNode0.putArray("UNKNOWN[?]");
      JsonParser jsonParser0 = arrayNode0.traverse((ObjectCodec) null);
      JsonMappingException jsonMappingException0 = JsonMappingException.from(jsonParser0, "BeanSerializer for ");
      assertNotNull(jsonMappingException0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Class<LineNumberReader> class0 = LineNumberReader.class;
      JsonMappingException jsonMappingException0 = defaultDeserializationContext_Impl0.weirdKeyException(class0, (String) null, "IP_ADDRESS");
      String string0 = jsonMappingException0.getLocalizedMessage();
      assertEquals("Cannot deserialize Map key of type `java.io.LineNumberReader` from String [N/A]: IP_ADDRESS", string0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      JsonMappingException.Reference jsonMappingException_Reference0 = new JsonMappingException.Reference();
      SQLNonTransientException sQLNonTransientException0 = new SQLNonTransientException("", "");
      JsonMappingException jsonMappingException0 = JsonMappingException.wrapWithPath((Throwable) sQLNonTransientException0, jsonMappingException_Reference0);
      JsonMappingException.fromUnexpectedIOE(jsonMappingException0);
      String string0 = jsonMappingException_Reference0.getDescription();
      assertNotNull(string0);
      assertEquals((-1), jsonMappingException_Reference0.getIndex());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      SQLWarning sQLWarning0 = new SQLWarning();
      SQLNonTransientConnectionException sQLNonTransientConnectionException0 = new SQLNonTransientConnectionException(sQLWarning0);
      SQLTransientException sQLTransientException0 = new SQLTransientException((String) null, sQLNonTransientConnectionException0);
      byte[] byteArray0 = new byte[5];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0, (byte)94, 1446);
      BufferedInputStream bufferedInputStream0 = new BufferedInputStream(byteArrayInputStream0);
      JsonMappingException jsonMappingException0 = JsonMappingException.wrapWithPath((Throwable) sQLTransientException0, (Object) bufferedInputStream0, (-590));
      assertNotNull(jsonMappingException0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      JsonMappingException.Reference jsonMappingException_Reference0 = new JsonMappingException.Reference();
      SQLNonTransientException sQLNonTransientException0 = new SQLNonTransientException("", "");
      JsonMappingException jsonMappingException0 = JsonMappingException.wrapWithPath((Throwable) sQLNonTransientException0, jsonMappingException_Reference0);
      List<JsonMappingException.Reference> list0 = jsonMappingException0.getPath();
      assertEquals((-1), jsonMappingException_Reference0.getIndex());
      assertTrue(list0.contains(jsonMappingException_Reference0));
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      JsonMappingException jsonMappingException0 = new JsonMappingException("pXA-^?02x");
      List<JsonMappingException.Reference> list0 = jsonMappingException0.getPath();
      assertEquals(0, list0.size());
  }
}