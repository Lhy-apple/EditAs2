/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 13:26:49 GMT 2023
 */

package com.fasterxml.jackson.databind.node;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.annotation.JsonFormat;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.core.JsonPointer;
import com.fasterxml.jackson.core.JsonToken;
import com.fasterxml.jackson.core.ObjectCodec;
import com.fasterxml.jackson.core.io.IOContext;
import com.fasterxml.jackson.core.json.UTF8JsonGenerator;
import com.fasterxml.jackson.core.util.BufferRecycler;
import com.fasterxml.jackson.core.util.ByteArrayBuilder;
import com.fasterxml.jackson.databind.BeanProperty;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.PropertyMetadata;
import com.fasterxml.jackson.databind.PropertyName;
import com.fasterxml.jackson.databind.SerializerProvider;
import com.fasterxml.jackson.databind.deser.BeanDeserializerFactory;
import com.fasterxml.jackson.databind.deser.DefaultDeserializationContext;
import com.fasterxml.jackson.databind.introspect.AnnotatedMember;
import com.fasterxml.jackson.databind.introspect.AnnotationMap;
import com.fasterxml.jackson.databind.jsontype.impl.AsPropertyTypeSerializer;
import com.fasterxml.jackson.databind.jsontype.impl.ClassNameIdResolver;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.BigIntegerNode;
import com.fasterxml.jackson.databind.node.BinaryNode;
import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import com.fasterxml.jackson.databind.node.JsonNodeType;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.fasterxml.jackson.databind.node.POJONode;
import com.fasterxml.jackson.databind.node.TextNode;
import com.fasterxml.jackson.databind.node.ValueNode;
import com.fasterxml.jackson.databind.ser.DefaultSerializerProvider;
import com.fasterxml.jackson.databind.type.SimpleType;
import com.fasterxml.jackson.databind.type.TypeFactory;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.math.BigDecimal;
import java.time.ZoneId;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.Spliterator;
import java.util.TreeSet;
import java.util.Vector;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.io.MockFileOutputStream;
import org.evosuite.runtime.testdata.EvoSuiteFile;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class ObjectNode_ESTest extends ObjectNode_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory(true);
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0);
      Iterator<String> iterator0 = objectNode0.fieldNames();
      assertNotNull(iterator0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory();
      HashMap<String, JsonNode> hashMap0 = new HashMap<String, JsonNode>();
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0, hashMap0);
      JsonNode jsonNode0 = objectNode0.putAll(objectNode0);
      assertEquals("", jsonNode0.asText());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory();
      HashMap<String, JsonNode> hashMap0 = new HashMap<String, JsonNode>();
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0, hashMap0);
      ObjectNode objectNode1 = objectNode0.putNull((String) null);
      assertFalse(objectNode1.isBigDecimal());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory();
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0);
      objectNode0.hashCode();
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      ArrayNode arrayNode0 = objectMapper0.createArrayNode();
      ObjectNode objectNode0 = arrayNode0.insertObject(2147483645);
      JsonPointer jsonPointer0 = JsonPointer.valueOf((String) null);
      JsonNode jsonNode0 = objectNode0._at(jsonPointer0);
      assertNull(jsonNode0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory();
      HashMap<String, JsonNode> hashMap0 = new HashMap<String, JsonNode>();
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0, hashMap0);
      ObjectNode objectNode1 = objectNode0.removeAll();
      assertFalse(objectNode1.isBigDecimal());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory();
      ObjectNode objectNode0 = jsonNodeFactory0.objectNode();
      Spliterator<JsonNode> spliterator0 = objectNode0.spliterator();
      assertNotNull(spliterator0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory();
      HashMap<String, JsonNode> hashMap0 = new HashMap<String, JsonNode>();
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0, hashMap0);
      JsonNode jsonNode0 = objectNode0.without("g]EIQsrVe{!k5\"");
      assertFalse(jsonNode0.isFloatingPointNumber());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ArrayNode arrayNode0 = new ArrayNode(jsonNodeFactory0);
      ObjectNode objectNode0 = arrayNode0.addObject();
      JsonToken jsonToken0 = objectNode0.asToken();
      assertFalse(jsonToken0.isScalarValue());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory(true);
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0);
      ObjectNode objectNode1 = objectNode0.putObject("z/czp;<");
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0.CHAR_CONCAT_BUFFER, true);
      JsonFactory jsonFactory0 = new JsonFactory();
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      ObjectMapper objectMapper0 = new ObjectMapper(jsonFactory0, defaultSerializerProvider_Impl0, defaultDeserializationContext_Impl0);
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream("Could not find constructor with ");
      UTF8JsonGenerator uTF8JsonGenerator0 = new UTF8JsonGenerator(iOContext0, 0, objectMapper0, mockFileOutputStream0);
      Class<TextNode> class0 = TextNode.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      ClassNameIdResolver classNameIdResolver0 = new ClassNameIdResolver(simpleType0, typeFactory0);
      PropertyName propertyName0 = PropertyName.USE_DEFAULT;
      AnnotationMap annotationMap0 = new AnnotationMap();
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_OPTIONAL;
      BeanProperty.Std beanProperty_Std0 = new BeanProperty.Std(propertyName0, simpleType0, propertyName0, annotationMap0, (AnnotatedMember) null, propertyMetadata0);
      AsPropertyTypeSerializer asPropertyTypeSerializer0 = new AsPropertyTypeSerializer(classNameIdResolver0, beanProperty_Std0, "+@4v`y%/:^n]U");
      objectNode0.serializeWithType(uTF8JsonGenerator0, defaultSerializerProvider_Impl0, asPropertyTypeSerializer0);
      assertNotSame(objectNode0, objectNode1);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ArrayNode arrayNode0 = new ArrayNode(jsonNodeFactory0);
      ObjectNode objectNode0 = arrayNode0.addObject();
      ObjectNode objectNode1 = objectNode0.put((String) null, (int) (byte) (-1));
      assertFalse(objectNode1.isFloatingPointNumber());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory();
      HashMap<String, JsonNode> hashMap0 = new HashMap<String, JsonNode>();
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0, hashMap0);
      String[] stringArray0 = new String[3];
      ObjectNode objectNode1 = objectNode0.retain(stringArray0);
      assertEquals("", objectNode1.asText());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory();
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0);
      HashMap<String, BigIntegerNode> hashMap0 = new HashMap<String, BigIntegerNode>();
      JsonNode jsonNode0 = objectNode0.putAll((Map<String, ? extends JsonNode>) hashMap0);
      assertSame(objectNode0, jsonNode0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      ObjectNode objectNode0 = new ObjectNode((JsonNodeFactory) null);
      boolean boolean0 = objectNode0.isValueNode();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory();
      ObjectNode objectNode0 = jsonNodeFactory0.objectNode();
      TreeSet<String> treeSet0 = new TreeSet<String>();
      ObjectNode objectNode1 = objectNode0.remove((Collection<String>) treeSet0);
      assertNull(objectNode1.textValue());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0);
      JsonNode jsonNode0 = objectNode0.remove("' has value that is not of type ArrayNode (but ");
      assertNull(jsonNode0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.withExactBigDecimals(true);
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0);
      JsonNode jsonNode0 = objectNode0.get((-2051104968));
      assertNull(jsonNode0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory();
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0);
      ObjectNode objectNode1 = objectNode0.put(" entries; o have ", (-2434L));
      assertFalse(objectNode1.isFloatingPointNumber());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory();
      HashMap<String, JsonNode> hashMap0 = new HashMap<String, JsonNode>();
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0, hashMap0);
      ObjectNode objectNode1 = objectNode0.put("J%yx@]z", (-1824.0));
      assertFalse(objectNode1.isFloatingPointNumber());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory();
      HashMap<String, JsonNode> hashMap0 = new HashMap<String, JsonNode>();
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0, hashMap0);
      ObjectNode objectNode1 = objectNode0.deepCopy();
      boolean boolean0 = objectNode1.equals(objectNode0);
      assertNotSame(objectNode1, objectNode0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory(false);
      ObjectNode objectNode0 = jsonNodeFactory0.objectNode();
      ObjectNode objectNode1 = objectNode0.put("com.fasterxml.jackson.databind.deser.impl.UnwrappedPropertyHandler", false);
      assertFalse(objectNode1.isShort());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0);
      Set<String> set0 = ZoneId.getAvailableZoneIds();
      ObjectNode objectNode1 = objectNode0.without((Collection<String>) set0);
      assertFalse(objectNode1.isFloat());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.withExactBigDecimals(true);
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0);
      JsonNode jsonNode0 = objectNode0.path((-2051104968));
      assertFalse(jsonNode0.isInt());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory();
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0);
      Iterator<Map.Entry<String, JsonNode>> iterator0 = (Iterator<Map.Entry<String, JsonNode>>)objectNode0.fields();
      assertNotNull(iterator0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory();
      HashMap<String, JsonNode> hashMap0 = new HashMap<String, JsonNode>();
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0, hashMap0);
      ArrayNode arrayNode0 = objectNode0.putArray(">?!}");
      assertEquals(JsonNodeType.ARRAY, arrayNode0.getNodeType());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory();
      HashMap<String, JsonNode> hashMap0 = new HashMap<String, JsonNode>();
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0, hashMap0);
      POJONode pOJONode0 = new POJONode(hashMap0);
      ObjectNode objectNode1 = objectNode0._put("V", pOJONode0);
      ObjectNode objectNode2 = objectNode0.deepCopy();
      assertEquals(1, objectNode2.size());
      assertNotSame(objectNode2, objectNode1);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.withExactBigDecimals(true);
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0);
      JsonNode jsonNode0 = objectNode0.path(" (expected type: ");
      assertNull(jsonNode0.numberType());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory(true);
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0);
      objectNode0.put("Z}I<eFzUK<]`K~x,*gK", (Double) null);
      JsonNode jsonNode0 = objectNode0.path("Z}I<eFzUK<]`K~x,*gK");
      assertEquals(JsonToken.VALUE_NULL, jsonNode0.asToken());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory();
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0);
      Short short0 = new Short((short)636);
      objectNode0.put(">Z", short0);
      // Undeclared exception!
      try { 
        objectNode0.with(">Z");
        fail("Expecting exception: UnsupportedOperationException");
      
      } catch(UnsupportedOperationException e) {
         //
         // Property '>Z' has value that is not of type ObjectNode (but com.fasterxml.jackson.databind.node.ShortNode)
         //
         verifyException("com.fasterxml.jackson.databind.node.ObjectNode", e);
      }
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory(true);
      HashMap<String, JsonNode> hashMap0 = new HashMap<String, JsonNode>();
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0, hashMap0);
      objectNode0.set("l7pjKP*S/wQ", objectNode0);
      ObjectNode objectNode1 = objectNode0.with("l7pjKP*S/wQ");
      assertEquals(1, objectNode1.size());
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory(true);
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0);
      ArrayNode arrayNode0 = jsonNodeFactory0.arrayNode();
      ObjectNode objectNode1 = objectNode0._put("^&NVl%GG}fTP", arrayNode0);
      ArrayNode arrayNode1 = objectNode1.withArray("^&NVl%GG}fTP");
      assertSame(arrayNode1, arrayNode0);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory();
      HashMap<String, JsonNode> hashMap0 = new HashMap<String, JsonNode>();
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0, hashMap0);
      objectNode0.put("", (JsonNode) objectNode0);
      JsonNode jsonNode0 = objectNode0.findValue("");
      assertNull(jsonNode0.textValue());
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory();
      HashMap<String, JsonNode> hashMap0 = new HashMap<String, JsonNode>();
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0, hashMap0);
      ObjectNode objectNode1 = objectNode0.putObject("XZ(b%>OF&122Wxog");
      JsonNode jsonNode0 = objectNode0.findValue("");
      assertNull(jsonNode0);
      assertNotSame(objectNode0, objectNode1);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory();
      HashMap<String, JsonNode> hashMap0 = new HashMap<String, JsonNode>();
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0, hashMap0);
      HashMap<String, ObjectNode> hashMap1 = new HashMap<String, ObjectNode>();
      hashMap1.put("l:;c$!PbX$IO%7PX", objectNode0);
      ObjectNode objectNode1 = objectNode0.with("");
      objectNode1.setAll((Map<String, ? extends JsonNode>) hashMap1);
      JsonNode jsonNode0 = objectNode1.findValue("");
      assertEquals(1, objectNode0.size());
      assertNotSame(objectNode0, jsonNode0);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.withExactBigDecimals(true);
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0);
      Long long0 = new Long(0L);
      ObjectNode objectNode1 = objectNode0.put("4bN 3+", long0);
      ArrayList<JsonNode> arrayList0 = new ArrayList<JsonNode>();
      List<JsonNode> list0 = objectNode1.findValues("' has value that is not of type ArrayNode (but ", (List<JsonNode>) arrayList0);
      assertTrue(list0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.withExactBigDecimals(true);
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0);
      objectNode0.put("' has value that is not of type ArrayNode (but ", 1.0F);
      ArrayList<JsonNode> arrayList0 = new ArrayList<JsonNode>();
      objectNode0.findValues("' has value that is not of type ArrayNode (but ", (List<JsonNode>) arrayList0);
      assertEquals(1, arrayList0.size());
      assertFalse(arrayList0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory(false);
      HashMap<String, JsonNode> hashMap0 = new HashMap<String, JsonNode>(1491, 1491);
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0, hashMap0);
      objectNode0.withArray("vmOcrP.hI L,/t");
      List<JsonNode> list0 = objectNode0.findValues("vmOcrP.hI L,/t", (List<JsonNode>) null);
      assertFalse(list0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.withExactBigDecimals(false);
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0);
      Double double0 = new Double(3507.43);
      ObjectNode objectNode1 = objectNode0.put("=", double0);
      List<String> list0 = objectNode1.findValuesAsText("=", (List<String>) null);
      assertEquals(1, list0.size());
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory();
      HashMap<String, JsonNode> hashMap0 = new HashMap<String, JsonNode>();
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0, hashMap0);
      POJONode pOJONode0 = new POJONode("{");
      objectNode0._put("{", pOJONode0);
      Vector<String> vector0 = new Vector<String>();
      List<String> list0 = objectNode0.findValuesAsText("J%yx@]z", (List<String>) vector0);
      assertEquals(0, list0.size());
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory();
      HashMap<String, JsonNode> hashMap0 = new HashMap<String, JsonNode>();
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0, hashMap0);
      ValueNode valueNode0 = jsonNodeFactory0.pojoNode(hashMap0);
      objectNode0._put("J%yx@]z", valueNode0);
      ArrayList<String> arrayList0 = new ArrayList<String>();
      objectNode0.findValuesAsText("J%yx@]z", (List<String>) arrayList0);
      assertFalse(arrayList0.isEmpty());
      assertEquals(1, arrayList0.size());
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      HashMap<String, JsonNode> hashMap0 = new HashMap<String, JsonNode>();
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0, hashMap0);
      ObjectNode objectNode1 = objectNode0.putObject("XZ(b%>OF&122Wxog");
      ObjectNode objectNode2 = objectNode0.findParent("");
      assertNotSame(objectNode0, objectNode1);
      assertNull(objectNode2);
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      HashMap<String, JsonNode> hashMap0 = new HashMap<String, JsonNode>();
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0, hashMap0);
      ObjectNode objectNode1 = objectNode0.putObject("XZ(b%>OF&122Wxog");
      objectNode1.put("", "~`rKaYXC%=v");
      objectNode0.findParent("");
      assertEquals(1, objectNode0.size());
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory();
      HashMap<String, JsonNode> hashMap0 = new HashMap<String, JsonNode>();
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0, hashMap0);
      objectNode0.putPOJO("J%yx@]z", hashMap0);
      List<JsonNode> list0 = objectNode0.findParents("J%yx@]z");
      assertEquals(1, list0.size());
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory();
      HashMap<String, JsonNode> hashMap0 = new HashMap<String, JsonNode>();
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0, hashMap0);
      ObjectNode objectNode1 = objectNode0.put("' has value that is not of type ArrayNode (but ", (Double) null);
      List<JsonNode> list0 = objectNode0.findParents("J%yx@]z");
      // Undeclared exception!
      try { 
        objectNode1.findParents("' has value that is not of type ArrayNode (but ", list0);
        fail("Expecting exception: UnsupportedOperationException");
      
      } catch(UnsupportedOperationException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("java.util.AbstractList", e);
      }
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory();
      HashMap<String, JsonNode> hashMap0 = new HashMap<String, JsonNode>();
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0, hashMap0);
      ValueNode valueNode0 = jsonNodeFactory0.pojoNode(hashMap0);
      ObjectNode objectNode1 = objectNode0._put("J%yx@]z", valueNode0);
      JsonFactory jsonFactory0 = new JsonFactory((ObjectCodec) null);
      ByteArrayBuilder byteArrayBuilder0 = new ByteArrayBuilder(2);
      OutputStreamWriter outputStreamWriter0 = new OutputStreamWriter(byteArrayBuilder0);
      JsonGenerator jsonGenerator0 = jsonFactory0.createGenerator((Writer) outputStreamWriter0);
      ObjectMapper objectMapper0 = new ObjectMapper();
      SerializerProvider serializerProvider0 = objectMapper0.getSerializerProvider();
      // Undeclared exception!
      try { 
        objectNode1.serialize(jsonGenerator0, serializerProvider0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // No ObjectCodec defined for the generator, can only serialize simple wrapper types (type passed java.util.HashMap)
         //
         verifyException("com.fasterxml.jackson.core.JsonGenerator", e);
      }
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.withExactBigDecimals(false);
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0);
      JsonNode jsonNode0 = objectNode0.set("!1x[VizU", (JsonNode) null);
      assertNull(jsonNode0.numberType());
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory();
      ObjectNode objectNode0 = jsonNodeFactory0.objectNode();
      HashMap<String, BinaryNode> hashMap0 = new HashMap<String, BinaryNode>();
      hashMap0.put("FAIL_ON_UNRESOLVED_OBJECT_IDS", (BinaryNode) null);
      objectNode0.setAll((Map<String, ? extends JsonNode>) hashMap0);
      assertEquals(1, objectNode0.size());
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory();
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0);
      JsonNode jsonNode0 = objectNode0.replace("v5|h,6J", objectNode0);
      assertNull(jsonNode0);
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory();
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0);
      JsonNode jsonNode0 = objectNode0.replace("|'O0", (JsonNode) null);
      assertNull(jsonNode0);
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory();
      HashMap<String, JsonNode> hashMap0 = new HashMap<String, JsonNode>();
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0, hashMap0);
      JsonNode jsonNode0 = objectNode0.put("@d.e]:r@+L8vI7-k", (JsonNode) null);
      assertNull(jsonNode0);
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory();
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0);
      ObjectNode objectNode1 = objectNode0.put("swB^x", (Short) null);
      assertFalse(objectNode1.isDouble());
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.withExactBigDecimals(true);
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0);
      Integer integer0 = Integer.getInteger("", (-1839));
      ObjectNode objectNode1 = objectNode0.put("", integer0);
      assertEquals(JsonNodeType.OBJECT, objectNode1.getNodeType());
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory();
      HashMap<String, JsonNode> hashMap0 = new HashMap<String, JsonNode>();
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0, hashMap0);
      ObjectNode objectNode1 = objectNode0.put("w%", (Integer) null);
      assertFalse(objectNode1.isLong());
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory();
      HashMap<String, JsonNode> hashMap0 = new HashMap<String, JsonNode>();
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0, hashMap0);
      ObjectNode objectNode1 = objectNode0.put("includeAs can not be null", (Long) null);
      assertFalse(objectNode1.isFloatingPointNumber());
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory();
      HashMap<String, JsonNode> hashMap0 = new HashMap<String, JsonNode>();
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0, hashMap0);
      Float float0 = new Float((-9.223372036854776E18));
      ObjectNode objectNode1 = objectNode0.put("CG^1')`5?[", float0);
      assertFalse(objectNode1.isBigDecimal());
  }

  @Test(timeout = 4000)
  public void test55()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory();
      HashMap<String, JsonNode> hashMap0 = new HashMap<String, JsonNode>();
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0, hashMap0);
      ObjectNode objectNode1 = objectNode0.put("", (Float) null);
      assertFalse(objectNode1.booleanValue());
  }

  @Test(timeout = 4000)
  public void test56()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory(false);
      ObjectNode objectNode0 = jsonNodeFactory0.objectNode();
      BigDecimal bigDecimal0 = BigDecimal.ONE;
      ObjectNode objectNode1 = objectNode0.put("!QDd}ci", bigDecimal0);
      assertEquals(1, objectNode1.size());
  }

  @Test(timeout = 4000)
  public void test57()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory(true);
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0);
      ObjectNode objectNode1 = objectNode0.put(" entries; now have ", (BigDecimal) null);
      assertSame(objectNode0, objectNode1);
  }

  @Test(timeout = 4000)
  public void test58()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory(true);
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0);
      ObjectNode objectNode1 = objectNode0.put("Z}I<eFzUK<]`K~x,*gK", (String) null);
      assertEquals(JsonToken.START_OBJECT, objectNode1.asToken());
  }

  @Test(timeout = 4000)
  public void test59()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.withExactBigDecimals(false);
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0);
      Boolean boolean0 = new Boolean("^.v27(&b?");
      ObjectNode objectNode1 = objectNode0.put("3d7&;;\u0006Ir!", boolean0);
      assertSame(objectNode0, objectNode1);
  }

  @Test(timeout = 4000)
  public void test60()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.withExactBigDecimals(true);
      HashMap<String, JsonNode> hashMap0 = new HashMap<String, JsonNode>();
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0, hashMap0);
      ObjectNode objectNode1 = objectNode0.put("", (Boolean) null);
      assertFalse(objectNode1.isInt());
  }

  @Test(timeout = 4000)
  public void test61()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory();
      HashMap<String, JsonNode> hashMap0 = new HashMap<String, JsonNode>();
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0, hashMap0);
      byte[] byteArray0 = new byte[3];
      ObjectNode objectNode1 = objectNode0.put("~$bC0[rk])Cg~", byteArray0);
      assertSame(objectNode1, objectNode0);
  }

  @Test(timeout = 4000)
  public void test62()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory();
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0);
      ObjectNode objectNode1 = objectNode0.put("", (byte[]) null);
      assertFalse(objectNode1.isLong());
  }

  @Test(timeout = 4000)
  public void test63()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory(false);
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0);
      boolean boolean0 = objectNode0.equals(objectNode0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test64()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory();
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0);
      boolean boolean0 = objectNode0.equals((Object) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test65()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory(true);
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0);
      JsonFormat.Shape jsonFormat_Shape0 = JsonFormat.Shape.BOOLEAN;
      boolean boolean0 = objectNode0.equals(jsonFormat_Shape0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test66()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.withExactBigDecimals(true);
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0);
      Short short0 = new Short((short)636);
      objectNode0.put("", short0);
      ObjectNode objectNode1 = objectNode0.put("; expected Class<ValueInstantiator>", (short) (-1959));
      String string0 = objectNode1.toString();
      assertEquals("{\"\":636,\"; expected Class<ValueInstantiator>\":-1959}", string0);
  }
}
