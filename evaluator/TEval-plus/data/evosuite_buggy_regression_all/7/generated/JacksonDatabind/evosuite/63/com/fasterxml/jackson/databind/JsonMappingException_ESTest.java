/*
 * This file was automatically generated by EvoSuite
 * Sat Jul 29 19:05:02 GMT 2023
 */

package com.fasterxml.jackson.databind;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.core.JsonLocation;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.util.ByteArrayBuilder;
import com.fasterxml.jackson.core.util.JsonParserDelegate;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JsonMappingException;
import com.fasterxml.jackson.databind.SerializerProvider;
import com.fasterxml.jackson.databind.deser.BeanDeserializerFactory;
import com.fasterxml.jackson.databind.deser.DefaultDeserializationContext;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import com.fasterxml.jackson.databind.ser.DefaultSerializerProvider;
import java.io.InputStream;
import java.io.PipedReader;
import java.sql.BatchUpdateException;
import java.sql.SQLRecoverableException;
import java.sql.SQLTimeoutException;
import java.util.List;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.io.MockIOException;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class JsonMappingException_ESTest extends JsonMappingException_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      JsonMappingException.Reference jsonMappingException_Reference0 = new JsonMappingException.Reference();
      jsonMappingException_Reference0.setIndex((-737));
      assertEquals((-737), jsonMappingException_Reference0.getIndex());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      JsonMappingException.Reference jsonMappingException_Reference0 = new JsonMappingException.Reference();
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
      JsonMappingException.Reference jsonMappingException_Reference0 = new JsonMappingException.Reference();
      jsonMappingException_Reference0.setFieldName("{");
      assertEquals((-1), jsonMappingException_Reference0.getIndex());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      JsonMappingException.Reference jsonMappingException_Reference0 = new JsonMappingException.Reference((Object) null);
      assertEquals((-1), jsonMappingException_Reference0.getIndex());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      BatchUpdateException batchUpdateException0 = new BatchUpdateException();
      JsonMappingException jsonMappingException0 = JsonMappingException.wrapWithPath((Throwable) batchUpdateException0, (Object) batchUpdateException0, "lp+");
      jsonMappingException0.prependPath((Object) "lp+", 1269);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      PipedReader pipedReader0 = new PipedReader();
      JsonMappingException.Reference jsonMappingException_Reference0 = new JsonMappingException.Reference(pipedReader0, (-1019));
      jsonMappingException_Reference0.setDescription("d,,x_Yb");
      assertEquals((-1019), jsonMappingException_Reference0.getIndex());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      JsonMappingException.Reference jsonMappingException_Reference0 = new JsonMappingException.Reference((Object) null, "");
      String string0 = jsonMappingException_Reference0.getFieldName();
      assertNotNull(string0);
      assertEquals((-1), jsonMappingException_Reference0.getIndex());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      JsonMappingException.Reference jsonMappingException_Reference0 = new JsonMappingException.Reference();
      int int0 = jsonMappingException_Reference0.getIndex();
      assertEquals((-1), int0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      BatchUpdateException batchUpdateException0 = new BatchUpdateException();
      JsonMappingException jsonMappingException0 = JsonMappingException.wrapWithPath((Throwable) batchUpdateException0, (Object) batchUpdateException0, "rlN.BNOpN");
      String string0 = jsonMappingException0.getPathReference();
      assertEquals("java.sql.BatchUpdateException[\"rlN.BNOpN\"]", string0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      BatchUpdateException batchUpdateException0 = new BatchUpdateException();
      JsonMappingException jsonMappingException0 = new JsonMappingException("8S$P 6Hy)", (JsonLocation) null, batchUpdateException0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      JsonMappingException jsonMappingException0 = JsonMappingException.from((JsonGenerator) null, "f.x]|^2e");
      assertNotNull(jsonMappingException0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      BatchUpdateException batchUpdateException0 = new BatchUpdateException();
      // Undeclared exception!
      try { 
        JsonMappingException.from((DeserializationContext) null, "", (Throwable) batchUpdateException0);
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
      // Undeclared exception!
      try { 
        JsonMappingException.from((DeserializationContext) null, "cv23@l<&FW$>");
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.JsonMappingException", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      JsonMappingException jsonMappingException0 = new JsonMappingException((String) null);
      Object object0 = jsonMappingException0.getProcessor();
      assertNull(object0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      BatchUpdateException batchUpdateException0 = new BatchUpdateException();
      JsonMappingException jsonMappingException0 = JsonMappingException.from((JsonGenerator) null, "com.fasterxml.jackson.databind.util.RawValue", (Throwable) batchUpdateException0);
      assertNotNull(jsonMappingException0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      SQLTimeoutException sQLTimeoutException0 = new SQLTimeoutException("k?C$jEQZq|Q(D1|{");
      JsonMappingException jsonMappingException0 = JsonMappingException.wrapWithPath((Throwable) sQLTimeoutException0, (Object) sQLTimeoutException0, 3112);
      assertNotNull(jsonMappingException0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      JsonMappingException jsonMappingException0 = JsonMappingException.from((SerializerProvider) defaultSerializerProvider_Impl0, "");
      assertNotNull(jsonMappingException0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      MockIOException mockIOException0 = new MockIOException();
      // Undeclared exception!
      try { 
        JsonMappingException.from((SerializerProvider) null, (String) null, (Throwable) mockIOException0);
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
      BatchUpdateException batchUpdateException0 = new BatchUpdateException();
      JsonMappingException.Reference jsonMappingException_Reference0 = new JsonMappingException.Reference();
      JsonMappingException jsonMappingException0 = JsonMappingException.wrapWithPath((Throwable) batchUpdateException0, jsonMappingException_Reference0);
      SQLRecoverableException sQLRecoverableException0 = new SQLRecoverableException(jsonMappingException0);
      assertEquals((-1), jsonMappingException_Reference0.getIndex());
      assertEquals("java.sql.SQLRecoverableException: com.fasterxml.jackson.databind.JsonMappingException: (was java.sql.BatchUpdateException) (through reference chain: UNKNOWN[?])", sQLRecoverableException0.toString());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      BatchUpdateException batchUpdateException0 = new BatchUpdateException();
      JsonMappingException jsonMappingException0 = JsonMappingException.wrapWithPath((Throwable) batchUpdateException0, (Object) "SVC", "SVC");
      jsonMappingException0.prependPath((Object) "SVC", "SVC");
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      BatchUpdateException batchUpdateException0 = new BatchUpdateException();
      JsonMappingException jsonMappingException0 = new JsonMappingException("", batchUpdateException0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      JsonLocation jsonLocation0 = JsonLocation.NA;
      JsonMappingException jsonMappingException0 = new JsonMappingException("{m(", jsonLocation0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      ByteArrayBuilder byteArrayBuilder0 = new ByteArrayBuilder();
      JsonMappingException jsonMappingException0 = new JsonMappingException(byteArrayBuilder0, "#n2G", (JsonLocation) null);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      JsonMappingException jsonMappingException0 = new JsonMappingException("");
      String string0 = jsonMappingException0.getLocalizedMessage();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      BatchUpdateException batchUpdateException0 = new BatchUpdateException();
      JsonFactory jsonFactory0 = new JsonFactory();
      JsonParser jsonParser0 = jsonFactory0.createParser((InputStream) null);
      JsonMappingException jsonMappingException0 = JsonMappingException.from(jsonParser0, "com.fasterxml.jackon.databind.deser.eanDeserializer$BeanReferrijg", (Throwable) batchUpdateException0);
      JsonMappingException jsonMappingException1 = JsonMappingException.fromUnexpectedIOE(jsonMappingException0);
      assertNotSame(jsonMappingException1, jsonMappingException0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      BatchUpdateException batchUpdateException0 = new BatchUpdateException();
      // Undeclared exception!
      try { 
        JsonMappingException.wrapWithPath((Throwable) batchUpdateException0, (Object) batchUpdateException0, (String) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // Can not pass null fieldName
         //
         verifyException("com.fasterxml.jackson.databind.JsonMappingException$Reference", e);
      }
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Class<Throwable> class0 = Throwable.class;
      JsonMappingException jsonMappingException0 = defaultDeserializationContext_Impl0.instantiationException(class0, "%cv23@l<EFW$Y>");
      JsonMappingException.Reference jsonMappingException_Reference0 = new JsonMappingException.Reference(class0, "%cv23@l<EFW$Y>");
      jsonMappingException0.prependPath(jsonMappingException_Reference0);
      jsonMappingException0.prependPath(jsonMappingException_Reference0);
      assertEquals((-1), jsonMappingException_Reference0.getIndex());
      
      String string0 = jsonMappingException0._buildMessage();
      assertEquals("Can not construct instance of java.lang.Throwable: %cv23@l<EFW$Y> (through reference chain: java.lang.Throwable[\"%cv23@l<EFW$Y>\"]->java.lang.Throwable[\"%cv23@l<EFW$Y>\"])", string0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Object object0 = new Object();
      JsonMappingException.Reference jsonMappingException_Reference0 = new JsonMappingException.Reference(object0, 3044);
      String string0 = jsonMappingException_Reference0.toString();
      assertEquals("java.lang.Object[3044]", string0);
      assertNotNull(string0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory(false);
      ArrayNode arrayNode0 = new ArrayNode(jsonNodeFactory0);
      JsonParser jsonParser0 = arrayNode0.traverse();
      JsonParserDelegate jsonParserDelegate0 = new JsonParserDelegate(jsonParser0);
      JsonMappingException jsonMappingException0 = JsonMappingException.from((JsonParser) jsonParserDelegate0, "7k\"");
      assertNotNull(jsonMappingException0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Class<Throwable> class0 = Throwable.class;
      JsonMappingException jsonMappingException0 = defaultDeserializationContext_Impl0.instantiationException(class0, "%cv23@l<EFW$Y>");
      JsonMappingException.Reference jsonMappingException_Reference0 = new JsonMappingException.Reference(class0, "%cv23@l<EFW$Y>");
      JsonMappingException.wrapWithPath((Throwable) jsonMappingException0, jsonMappingException_Reference0);
      assertEquals((-1), jsonMappingException_Reference0.getIndex());
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      BatchUpdateException batchUpdateException0 = new BatchUpdateException("", "", (int[]) null);
      JsonMappingException jsonMappingException0 = JsonMappingException.wrapWithPath((Throwable) batchUpdateException0, (Object) batchUpdateException0, "");
      assertNotNull(jsonMappingException0);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      BatchUpdateException batchUpdateException0 = new BatchUpdateException();
      JsonMappingException.Reference jsonMappingException_Reference0 = new JsonMappingException.Reference();
      JsonMappingException jsonMappingException0 = JsonMappingException.wrapWithPath((Throwable) batchUpdateException0, jsonMappingException_Reference0);
      List<JsonMappingException.Reference> list0 = jsonMappingException0.getPath();
      assertEquals((-1), jsonMappingException_Reference0.getIndex());
      assertTrue(list0.contains(jsonMappingException_Reference0));
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      JsonMappingException jsonMappingException0 = new JsonMappingException("com.fasterxml.jackson.databind.util.RawValue");
      List<JsonMappingException.Reference> list0 = jsonMappingException0.getPath();
      assertEquals(0, list0.size());
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      JsonMappingException jsonMappingException0 = new JsonMappingException("EeVbr4E#Z\"pwWjX&H");
      jsonMappingException0._appendPathDesc((StringBuilder) null);
  }
}