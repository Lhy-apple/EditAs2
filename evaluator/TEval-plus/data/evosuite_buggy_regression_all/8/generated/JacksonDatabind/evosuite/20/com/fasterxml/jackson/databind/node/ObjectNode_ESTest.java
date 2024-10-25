/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 03:03:17 GMT 2023
 */

package com.fasterxml.jackson.databind.node;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.core.JsonEncoding;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonPointer;
import com.fasterxml.jackson.core.JsonToken;
import com.fasterxml.jackson.core.io.IOContext;
import com.fasterxml.jackson.core.json.WriterBasedJsonGenerator;
import com.fasterxml.jackson.core.util.BufferRecycler;
import com.fasterxml.jackson.core.util.JsonGeneratorDelegate;
import com.fasterxml.jackson.databind.BeanProperty;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.jsontype.impl.AsExistingPropertyTypeSerializer;
import com.fasterxml.jackson.databind.jsontype.impl.ClassNameIdResolver;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.BigIntegerNode;
import com.fasterxml.jackson.databind.node.BinaryNode;
import com.fasterxml.jackson.databind.node.FloatNode;
import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import com.fasterxml.jackson.databind.node.JsonNodeType;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.fasterxml.jackson.databind.ser.DefaultSerializerProvider;
import com.fasterxml.jackson.databind.type.SimpleType;
import com.fasterxml.jackson.databind.type.TypeFactory;
import java.io.ByteArrayOutputStream;
import java.io.OutputStream;
import java.math.BigDecimal;
import java.math.BigInteger;
import java.nio.charset.Charset;
import java.util.Collection;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.Spliterator;
import java.util.TreeSet;
import java.util.Vector;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.io.MockFile;
import org.evosuite.runtime.mock.java.io.MockPrintStream;
import org.evosuite.runtime.mock.java.io.MockPrintWriter;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class ObjectNode_ESTest extends ObjectNode_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory();
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0);
      boolean boolean0 = objectNode0.hasNonNull("com.fasterxml.jackson.databind.node.ObjectNode");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ObjectNode objectNode0 = jsonNodeFactory0.objectNode();
      // Undeclared exception!
      try { 
        objectNode0.putAll((ObjectNode) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.node.ObjectNode", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ObjectNode objectNode0 = jsonNodeFactory0.objectNode();
      Iterator<String> iterator0 = objectNode0.fieldNames();
      assertNotNull(iterator0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ObjectNode objectNode0 = jsonNodeFactory0.objectNode();
      ObjectNode objectNode1 = objectNode0.putNull("com.fasterxml.jackson.databind.type.TypeBindings");
      assertSame(objectNode0, objectNode1);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.withExactBigDecimals(true);
      HashMap<String, JsonNode> hashMap0 = new HashMap<String, JsonNode>();
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0, hashMap0);
      JsonNode jsonNode0 = hashMap0.remove((Object) objectNode0);
      assertNull(jsonNode0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory(true);
      ObjectNode objectNode0 = jsonNodeFactory0.objectNode();
      // Undeclared exception!
      try { 
        objectNode0._at((JsonPointer) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.node.ObjectNode", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.withExactBigDecimals(true);
      HashMap<String, JsonNode> hashMap0 = new HashMap<String, JsonNode>();
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0, hashMap0);
      ObjectNode objectNode1 = objectNode0.removeAll();
      assertSame(objectNode1, objectNode0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ObjectNode objectNode0 = jsonNodeFactory0.objectNode();
      Spliterator<JsonNode> spliterator0 = objectNode0.spliterator();
      assertNotNull(spliterator0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory();
      ObjectNode objectNode0 = jsonNodeFactory0.objectNode();
      JsonNode jsonNode0 = objectNode0.without("#5|7qMg]SEh,gAbbQQf");
      assertNull(jsonNode0.numberType());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0);
      JsonToken jsonToken0 = objectNode0.asToken();
      assertEquals(JsonToken.START_OBJECT, jsonToken0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ObjectNode objectNode0 = jsonNodeFactory0.objectNode();
      ObjectNode objectNode1 = objectNode0.putObject("i|cK|");
      List<String> list0 = objectNode0.findValuesAsText("Can not use FormatSchema of type ");
      assertNotSame(objectNode0, objectNode1);
      assertEquals(0, list0.size());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ObjectNode objectNode0 = jsonNodeFactory0.objectNode();
      // Undeclared exception!
      try { 
        objectNode0.retain((String[]) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("java.util.Objects", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ObjectNode objectNode0 = jsonNodeFactory0.objectNode();
      HashMap<String, FloatNode> hashMap0 = new HashMap<String, FloatNode>();
      JsonNode jsonNode0 = objectNode0.putAll((Map<String, ? extends JsonNode>) hashMap0);
      assertFalse(jsonNode0.booleanValue());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ObjectNode objectNode0 = jsonNodeFactory0.objectNode();
      Locale locale0 = Locale.CHINESE;
      Set<String> set0 = locale0.getUnicodeLocaleKeys();
      ObjectNode objectNode1 = objectNode0.remove((Collection<String>) set0);
      assertFalse(objectNode1.isIntegralNumber());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ObjectNode objectNode0 = jsonNodeFactory0.objectNode();
      JsonNode jsonNode0 = objectNode0.remove(":efq$IY{H1Pd(/3\"");
      assertNull(jsonNode0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0);
      boolean boolean0 = objectNode0.has((-1635045077));
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ObjectNode objectNode0 = jsonNodeFactory0.objectNode();
      Charset charset0 = Charset.defaultCharset();
      Set<String> set0 = charset0.aliases();
      ObjectNode objectNode1 = objectNode0.retain((Collection<String>) set0);
      assertFalse(objectNode1.isInt());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ObjectNode objectNode0 = jsonNodeFactory0.objectNode();
      ObjectNode objectNode1 = objectNode0.put("Y: i", 0.0);
      assertNull(objectNode1.textValue());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0);
      ObjectNode objectNode1 = new ObjectNode(jsonNodeFactory0);
      boolean boolean0 = objectNode1.equals(objectNode0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory();
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0);
      ObjectNode objectNode1 = objectNode0.put((String) null, true);
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, objectNode1, true);
      ObjectMapper objectMapper0 = new ObjectMapper();
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      MockPrintWriter mockPrintWriter0 = new MockPrintWriter(byteArrayOutputStream0, true);
      WriterBasedJsonGenerator writerBasedJsonGenerator0 = new WriterBasedJsonGenerator(iOContext0, 660, objectMapper0, mockPrintWriter0);
      JsonGeneratorDelegate jsonGeneratorDelegate0 = new JsonGeneratorDelegate(writerBasedJsonGenerator0);
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      Class<Locale.LanguageRange> class0 = Locale.LanguageRange.class;
      SimpleType simpleType0 = SimpleType.construct(class0);
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      ClassNameIdResolver classNameIdResolver0 = new ClassNameIdResolver(simpleType0, typeFactory0);
      AsExistingPropertyTypeSerializer asExistingPropertyTypeSerializer0 = new AsExistingPropertyTypeSerializer(classNameIdResolver0, (BeanProperty) null, (String) null);
      // Undeclared exception!
      try { 
        objectNode1.serializeWithType(jsonGeneratorDelegate0, defaultSerializerProvider_Impl0, asExistingPropertyTypeSerializer0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.core.json.WriterBasedJsonGenerator", e);
      }
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ObjectNode objectNode0 = jsonNodeFactory0.objectNode();
      TreeSet<String> treeSet0 = new TreeSet<String>();
      ObjectNode objectNode1 = objectNode0.without((Collection<String>) treeSet0);
      assertEquals(JsonToken.START_OBJECT, objectNode1.asToken());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory();
      ObjectNode objectNode0 = jsonNodeFactory0.objectNode();
      ObjectNode objectNode1 = objectNode0.put("RHPf3~^dw=w?=,-M(", (-1.0F));
      List<JsonNode> list0 = objectNode1.findValues("5#");
      assertEquals(0, list0.size());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory(false);
      ObjectNode objectNode0 = jsonNodeFactory0.objectNode();
      JsonNode jsonNode0 = objectNode0.path((int) (short)12552);
      assertFalse(jsonNode0.isBigInteger());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ObjectNode objectNode0 = jsonNodeFactory0.objectNode();
      JsonParser jsonParser0 = objectNode0.traverse();
      assertFalse(jsonParser0.hasCurrentToken());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory();
      ObjectNode objectNode0 = jsonNodeFactory0.objectNode();
      ArrayNode arrayNode0 = objectNode0.putArray("v,~|46yta$;)");
      assertFalse(arrayNode0.isDouble());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0);
      ObjectNode objectNode1 = objectNode0.putPOJO("", "");
      assertFalse(objectNode1.isDouble());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      HashMap<String, JsonNode> hashMap0 = new HashMap<String, JsonNode>();
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0, hashMap0);
      ObjectNode objectNode1 = objectNode0.deepCopy();
      objectNode1.withArray("RHPf3~^dw*w?=,nM(");
      hashMap0.put("Iw8az>g", objectNode1);
      JsonNode jsonNode0 = objectNode0.findValue("RHPf3~^dw*w?=,nM(");
      assertEquals(0, jsonNode0.size());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      HashMap<String, JsonNode> hashMap0 = new HashMap<String, JsonNode>();
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0, hashMap0);
      ObjectNode objectNode1 = objectNode0.put("Iw8az>g", false);
      ObjectNode objectNode2 = objectNode1.deepCopy();
      assertEquals(1, objectNode2.size());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      HashMap<String, JsonNode> hashMap0 = new HashMap<String, JsonNode>();
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory(false);
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0, hashMap0);
      JsonNode jsonNode0 = objectNode0.path("O,P+Qm2ZT%I!(&o");
      assertNull(jsonNode0.textValue());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      HashMap<String, JsonNode> hashMap0 = new HashMap<String, JsonNode>();
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory(false);
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0, hashMap0);
      objectNode0._put("' has value that is not of type ObjectNode (but ", objectNode0);
      JsonNode jsonNode0 = objectNode0.path("' has value that is not of type ObjectNode (but ");
      assertSame(jsonNode0, objectNode0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory();
      HashMap<String, JsonNode> hashMap0 = new HashMap<String, JsonNode>();
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0, hashMap0);
      ObjectNode objectNode1 = objectNode0.with("Internal error: unrecognized value of type ");
      ObjectNode objectNode2 = objectNode0.with("Internal error: unrecognized value of type ");
      assertSame(objectNode2, objectNode1);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0);
      BigInteger bigInteger0 = BigInteger.ONE;
      BigIntegerNode bigIntegerNode0 = new BigIntegerNode(bigInteger0);
      ObjectNode objectNode1 = objectNode0._put("", bigIntegerNode0);
      // Undeclared exception!
      try { 
        objectNode1.with("");
        fail("Expecting exception: UnsupportedOperationException");
      
      } catch(UnsupportedOperationException e) {
         //
         // Property '' has value that is not of type ObjectNode (but com.fasterxml.jackson.databind.node.BigIntegerNode)
         //
         verifyException("com.fasterxml.jackson.databind.node.ObjectNode", e);
      }
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory();
      HashMap<String, JsonNode> hashMap0 = new HashMap<String, JsonNode>();
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0, hashMap0);
      BigDecimal bigDecimal0 = BigDecimal.ONE;
      objectNode0.put((String) null, bigDecimal0);
      // Undeclared exception!
      try { 
        objectNode0.withArray((String) null);
        fail("Expecting exception: UnsupportedOperationException");
      
      } catch(UnsupportedOperationException e) {
         //
         // Property 'null' has value that is not of type ArrayNode (but com.fasterxml.jackson.databind.node.DecimalNode)
         //
         verifyException("com.fasterxml.jackson.databind.node.ObjectNode", e);
      }
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0);
      objectNode0.withArray("gxsw{:#}Vlp{?%hx");
      ArrayNode arrayNode0 = objectNode0.withArray("gxsw{:#}Vlp{?%hx");
      assertFalse(arrayNode0.isShort());
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory();
      HashMap<String, JsonNode> hashMap0 = new HashMap<String, JsonNode>();
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0, hashMap0);
      ObjectNode objectNode1 = objectNode0.put("", (-2147483648L));
      JsonNode jsonNode0 = objectNode1.findValue("'V[`=iC");
      assertNull(jsonNode0);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory();
      ObjectNode objectNode0 = jsonNodeFactory0.objectNode();
      ObjectNode objectNode1 = objectNode0.with("");
      List<JsonNode> list0 = objectNode0.findValues("");
      assertFalse(list0.isEmpty());
      assertNotSame(objectNode0, objectNode1);
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0);
      BigInteger bigInteger0 = BigInteger.ONE;
      BigIntegerNode bigIntegerNode0 = BigIntegerNode.valueOf(bigInteger0);
      Vector<JsonNode> vector0 = new Vector<JsonNode>();
      objectNode0.set("", bigIntegerNode0);
      objectNode0.findValues("", (List<JsonNode>) vector0);
      assertEquals(1, vector0.size());
      assertEquals("[1]", vector0.toString());
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.withExactBigDecimals(true);
      HashMap<String, JsonNode> hashMap0 = new HashMap<String, JsonNode>();
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0, hashMap0);
      List<String> list0 = objectNode0.findValuesAsText("h>-1.@v4Y+<;");
      objectNode0.put("h>-1.@v4Y+<;", (-1097));
      // Undeclared exception!
      try { 
        objectNode0.findValuesAsText("h>-1.@v4Y+<;", list0);
        fail("Expecting exception: UnsupportedOperationException");
      
      } catch(UnsupportedOperationException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("java.util.AbstractList", e);
      }
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory(false);
      HashMap<String, JsonNode> hashMap0 = new HashMap<String, JsonNode>();
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0, hashMap0);
      objectNode0.put("}", (short) (-253));
      List<String> list0 = objectNode0.findValuesAsText("}");
      assertFalse(list0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory();
      HashMap<String, JsonNode> hashMap0 = new HashMap<String, JsonNode>();
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0, hashMap0);
      ObjectNode objectNode1 = objectNode0.put("", (-2147483648L));
      ObjectNode objectNode2 = objectNode1.findParent("Internal error: unrecognized value of type ");
      assertNull(objectNode2);
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory();
      HashMap<String, JsonNode> hashMap0 = new HashMap<String, JsonNode>();
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0, hashMap0);
      ObjectNode objectNode1 = objectNode0.with("Internal error: unrecognized value of type ");
      objectNode1.set("", objectNode0);
      assertEquals(1, objectNode1.size());
      
      ObjectNode objectNode2 = objectNode1.findParent("Internal error: unrecognized value of type ");
      assertSame(objectNode0, objectNode2);
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory(false);
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0);
      Double double0 = new Double(593.8852851392427);
      ObjectNode objectNode1 = objectNode0.put("ukwku", double0);
      List<JsonNode> list0 = objectNode1.findParents("_^w");
      assertEquals(0, list0.size());
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0);
      Integer integer0 = new Integer(1119);
      ObjectNode objectNode1 = objectNode0.put("", integer0);
      List<JsonNode> list0 = objectNode0.findParents("");
      objectNode1.findParents("", list0);
      assertEquals(2, list0.size());
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory();
      ObjectNode objectNode0 = jsonNodeFactory0.objectNode();
      FloatNode floatNode0 = new FloatNode(860.99554F);
      ObjectNode objectNode1 = objectNode0._put(", can not serialize", floatNode0);
      JsonFactory jsonFactory0 = new JsonFactory();
      MockFile mockFile0 = new MockFile("JSON", "JSON");
      MockPrintStream mockPrintStream0 = new MockPrintStream(mockFile0);
      JsonEncoding jsonEncoding0 = JsonEncoding.UTF32_LE;
      JsonGenerator jsonGenerator0 = jsonFactory0.createGenerator((OutputStream) mockPrintStream0, jsonEncoding0);
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      objectNode1.serialize(jsonGenerator0, defaultSerializerProvider_Impl0);
      assertFalse(objectNode1.isLong());
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory();
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0);
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, objectNode0, true);
      ObjectMapper objectMapper0 = new ObjectMapper();
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      MockPrintWriter mockPrintWriter0 = new MockPrintWriter(byteArrayOutputStream0, true);
      WriterBasedJsonGenerator writerBasedJsonGenerator0 = new WriterBasedJsonGenerator(iOContext0, 660, objectMapper0, mockPrintWriter0);
      JsonGeneratorDelegate jsonGeneratorDelegate0 = new JsonGeneratorDelegate(writerBasedJsonGenerator0);
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      Class<Locale.LanguageRange> class0 = Locale.LanguageRange.class;
      SimpleType simpleType0 = SimpleType.construct(class0);
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      ClassNameIdResolver classNameIdResolver0 = new ClassNameIdResolver(simpleType0, typeFactory0);
      AsExistingPropertyTypeSerializer asExistingPropertyTypeSerializer0 = new AsExistingPropertyTypeSerializer(classNameIdResolver0, (BeanProperty) null, (String) null);
      objectNode0.serializeWithType(jsonGeneratorDelegate0, defaultSerializerProvider_Impl0, asExistingPropertyTypeSerializer0);
      assertEquals(0, objectNode0.size());
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0);
      JsonNode jsonNode0 = objectNode0.set("{", (JsonNode) null);
      assertEquals("", jsonNode0.asText());
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory();
      HashMap<String, JsonNode> hashMap0 = new HashMap<String, JsonNode>();
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0, hashMap0);
      ObjectNode objectNode1 = objectNode0.put("", (-2147483648L));
      JsonNode jsonNode0 = objectNode1.setAll((Map<String, ? extends JsonNode>) hashMap0);
      assertSame(jsonNode0, objectNode1);
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory(false);
      ArrayNode arrayNode0 = jsonNodeFactory0.arrayNode();
      ObjectNode objectNode0 = arrayNode0.insertObject((-1984397772));
      HashMap<String, BinaryNode> hashMap0 = new HashMap<String, BinaryNode>();
      hashMap0.put((String) null, (BinaryNode) null);
      objectNode0.setAll((Map<String, ? extends JsonNode>) hashMap0);
      assertEquals(1, objectNode0.size());
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0);
      JsonNode jsonNode0 = objectNode0.replace(")\"6^dz#($&yX", objectNode0);
      assertNull(jsonNode0);
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ObjectNode objectNode0 = jsonNodeFactory0.objectNode();
      ArrayNode arrayNode0 = objectNode0.arrayNode();
      JsonNode jsonNode0 = objectNode0.put("*\"u#aSW|w", (JsonNode) arrayNode0);
      assertNull(jsonNode0);
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0);
      JsonNode jsonNode0 = objectNode0.put("}|c", (JsonNode) null);
      assertNull(jsonNode0);
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ObjectNode objectNode0 = jsonNodeFactory0.objectNode();
      Short short0 = new Short((short) (-17169));
      ObjectNode objectNode1 = objectNode0.put("_", short0);
      assertEquals(JsonNodeType.OBJECT, objectNode1.getNodeType());
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ObjectNode objectNode0 = jsonNodeFactory0.objectNode();
      ObjectNode objectNode1 = objectNode0.put((String) null, (Short) null);
      assertFalse(objectNode1.isFloatingPointNumber());
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory();
      ObjectNode objectNode0 = jsonNodeFactory0.objectNode();
      ObjectNode objectNode1 = objectNode0.put("", (Integer) null);
      assertNull(objectNode1.numberType());
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ObjectNode objectNode0 = jsonNodeFactory0.objectNode();
      Long long0 = new Long((-583L));
      ObjectNode objectNode1 = objectNode0.put("' has value that is not of type ObjectNode (but ", long0);
      assertFalse(objectNode1.booleanValue());
  }

  @Test(timeout = 4000)
  public void test55()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0);
      ObjectNode objectNode1 = objectNode0.put("k)aL&<_+>%=Q$~d9", (Long) null);
      assertFalse(objectNode1.isBigInteger());
  }

  @Test(timeout = 4000)
  public void test56()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory(false);
      HashMap<String, JsonNode> hashMap0 = new HashMap<String, JsonNode>();
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0, hashMap0);
      Float float0 = new Float((double) (short) (-253));
      ObjectNode objectNode1 = objectNode0.put("}", float0);
      assertNull(objectNode1.textValue());
  }

  @Test(timeout = 4000)
  public void test57()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory();
      ObjectNode objectNode0 = jsonNodeFactory0.objectNode();
      ObjectNode objectNode1 = objectNode0.put("}", (Float) null);
      assertFalse(objectNode1.isShort());
  }

  @Test(timeout = 4000)
  public void test58()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      HashMap<String, JsonNode> hashMap0 = new HashMap<String, JsonNode>();
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0, hashMap0);
      ObjectNode objectNode1 = objectNode0.put("rc)0Z(c&;r~^a6$Mz<i", (Double) null);
      assertSame(objectNode1, objectNode0);
  }

  @Test(timeout = 4000)
  public void test59()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory();
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0);
      ObjectNode objectNode1 = objectNode0.put("_{", (BigDecimal) null);
      assertFalse(objectNode1.isDouble());
  }

  @Test(timeout = 4000)
  public void test60()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.withExactBigDecimals(true);
      HashMap<String, JsonNode> hashMap0 = new HashMap<String, JsonNode>();
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0, hashMap0);
      ObjectNode objectNode1 = objectNode0.put("' has value that is not of type ObjectNode (but ", "' has value that is not of type ObjectNode (but ");
      assertFalse(objectNode1.isInt());
  }

  @Test(timeout = 4000)
  public void test61()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory();
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0);
      ObjectNode objectNode1 = objectNode0.put((String) null, (String) null);
      assertEquals(JsonToken.START_OBJECT, objectNode1.asToken());
  }

  @Test(timeout = 4000)
  public void test62()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0);
      Boolean boolean0 = Boolean.FALSE;
      ObjectNode objectNode1 = objectNode0.put("V", boolean0);
      assertEquals("", objectNode1.asText());
  }

  @Test(timeout = 4000)
  public void test63()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ObjectNode objectNode0 = jsonNodeFactory0.objectNode();
      ObjectNode objectNode1 = objectNode0.put("Lus`G}37E3;", (Boolean) null);
      assertEquals(JsonToken.START_OBJECT, objectNode1.asToken());
  }

  @Test(timeout = 4000)
  public void test64()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ObjectNode objectNode0 = jsonNodeFactory0.objectNode();
      byte[] byteArray0 = new byte[0];
      ObjectNode objectNode1 = objectNode0.put("fQ+D)%D<", byteArray0);
      assertFalse(objectNode1.isDouble());
  }

  @Test(timeout = 4000)
  public void test65()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ObjectNode objectNode0 = jsonNodeFactory0.objectNode();
      ObjectNode objectNode1 = objectNode0.put("RHPf3~^dw*w?=,nM(", (byte[]) null);
      assertEquals(JsonToken.START_OBJECT, objectNode1.asToken());
  }

  @Test(timeout = 4000)
  public void test66()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0);
      boolean boolean0 = objectNode0.equals(objectNode0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test67()  throws Throwable  {
      HashMap<String, JsonNode> hashMap0 = new HashMap<String, JsonNode>();
      ObjectNode objectNode0 = new ObjectNode((JsonNodeFactory) null, hashMap0);
      boolean boolean0 = objectNode0.equals((Object) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test68()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory(false);
      ObjectNode objectNode0 = jsonNodeFactory0.objectNode();
      FloatNode floatNode0 = new FloatNode((short)12552);
      boolean boolean0 = objectNode0.equals(floatNode0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test69()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      HashMap<String, JsonNode> hashMap0 = new HashMap<String, JsonNode>();
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0, hashMap0);
      objectNode0.replace(") has not properly overridden method 'withAdditionalSerializers': can not instantiate subtype with ", (JsonNode) null);
      ObjectNode objectNode1 = objectNode0.put("{@d7`,31#nf87", (-1111));
      String string0 = objectNode1.toString();
      assertEquals("{\") has not properly overridden method 'withAdditionalSerializers': can not instantiate subtype with \":null,\"{@d7`,31#nf87\":-1111}", string0);
  }
}
