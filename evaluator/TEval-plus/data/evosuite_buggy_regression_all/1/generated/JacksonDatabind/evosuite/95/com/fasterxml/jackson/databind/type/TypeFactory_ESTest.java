/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 13:42:36 GMT 2023
 */

package com.fasterxml.jackson.databind.type;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.AnnotationIntrospector;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.JsonSerializer;
import com.fasterxml.jackson.databind.MapperFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.type.ArrayType;
import com.fasterxml.jackson.databind.type.ClassStack;
import com.fasterxml.jackson.databind.type.CollectionLikeType;
import com.fasterxml.jackson.databind.type.CollectionType;
import com.fasterxml.jackson.databind.type.MapLikeType;
import com.fasterxml.jackson.databind.type.MapType;
import com.fasterxml.jackson.databind.type.PlaceholderForType;
import com.fasterxml.jackson.databind.type.ReferenceType;
import com.fasterxml.jackson.databind.type.SimpleType;
import com.fasterxml.jackson.databind.type.TypeBindings;
import com.fasterxml.jackson.databind.type.TypeFactory;
import com.fasterxml.jackson.databind.type.TypeModifier;
import com.fasterxml.jackson.databind.type.TypeParser;
import com.fasterxml.jackson.databind.util.LRUMap;
import java.lang.reflect.Array;
import java.lang.reflect.ParameterizedType;
import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.Properties;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.concurrent.atomic.AtomicReference;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class TypeFactory_ESTest extends TypeFactory_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<ReferenceType> class0 = ReferenceType.class;
      Class<MapperFeature> class1 = MapperFeature.class;
      JavaType[] javaTypeArray0 = typeFactory0.findTypeParameters(class0, class1);
      assertEquals(0, javaTypeArray0.length);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<MapType> class0 = MapType.class;
      JavaType javaType0 = typeFactory0.constructSimpleType(class0, class0, (JavaType[]) null);
      assertTrue(javaType0.isFinal());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<HashSet> class0 = HashSet.class;
      CollectionType collectionType0 = typeFactory0.constructRawCollectionType(class0);
      assertFalse(collectionType0.hasValueHandler());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      TypeReference<ObjectMapper.DefaultTyping> typeReference0 = (TypeReference<ObjectMapper.DefaultTyping>) mock(TypeReference.class, new ViolatedAssumptionAnswer());
      doReturn((Type) null).when(typeReference0).getType();
      String string0 = objectMapper0.writeValueAsString(typeReference0);
      assertEquals("{\"mockitoInterceptor\":{\"serializationSupport\":{},\"mockHandler\":{\"invocationContainer\":{\"invocationForStubbing\":null,\"stubbingsAscending\":[],\"invocations\":[],\"stubbingsDescending\":[]},\"mockSettings\":{\"outerClassInstance\":null,\"constructorArgs\":null,\"lenient\":false,\"stripAnnotations\":false,\"stubOnly\":false,\"typeToMock\":\"com.fasterxml.jackson.core.type.TypeReference\",\"spiedInstance\":null,\"name\":null,\"invocationListeners\":[{\"copyOfMethodDescriptors\":[]}],\"stubbingLookupListeners\":[],\"verificationStartedListeners\":[],\"extraInterfaces\":[],\"mockName\":{\"default\":true},\"serializableMode\":\"NONE\",\"defaultAnswer\":\"RETURNS_DEFAULTS\",\"serializable\":false,\"usingConstructor\":false}}},\"type\":null}", string0);
      
      TypeFactory.defaultInstance();
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<Object> class0 = Object.class;
      Class<DeserializationFeature> class1 = DeserializationFeature.class;
      Class<Integer>[] classArray0 = (Class<Integer>[]) Array.newInstance(Class.class, 0);
      JavaType javaType0 = typeFactory0.constructParametrizedType(class0, class1, classArray0);
      assertTrue(javaType0.isJavaLangObject());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      Class<Object> class0 = Object.class;
      ObjectReader objectReader0 = objectMapper0.readerFor(class0);
      TypeFactory typeFactory0 = objectReader0.getTypeFactory();
      Class<Properties> class1 = Properties.class;
      // Undeclared exception!
      try { 
        typeFactory0.constructRawMapType(class1);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Can not create TypeBindings for class java.util.Properties with 2 type parameters: class expects 0
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeBindings", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<Properties> class0 = Properties.class;
      MapLikeType mapLikeType0 = typeFactory0.constructMapLikeType(class0, class0, class0);
      Class<SerializationFeature> class1 = SerializationFeature.class;
      // Undeclared exception!
      try { 
        typeFactory0.constructSpecializedType(mapLikeType0, class1);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Class com.fasterxml.jackson.databind.SerializationFeature not subtype of [map type; class java.util.Properties, [simple type, class java.lang.String] -> [simple type, class java.lang.String]]
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<Properties> class0 = Properties.class;
      Class<Integer> class1 = Integer.class;
      JavaType javaType0 = typeFactory0.constructParametrizedType(class0, class1, (JavaType[]) null);
      // Undeclared exception!
      try { 
        typeFactory0.constructGeneralizedType(javaType0, class1);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Class java.lang.Integer not a super-type of [map type; class java.util.Properties, [simple type, class java.lang.String] -> [simple type, class java.lang.String]]
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      typeFactory0.clearCache();
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<HashSet> class0 = HashSet.class;
      CollectionLikeType collectionLikeType0 = typeFactory0.constructRawCollectionLikeType(class0);
      assertTrue(collectionLikeType0.isConcrete());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<Properties> class0 = Properties.class;
      CollectionLikeType collectionLikeType0 = typeFactory0.constructCollectionLikeType(class0, class0);
      assertFalse(collectionLikeType0.isInterface());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<Integer> class0 = Integer.class;
      ArrayType arrayType0 = typeFactory0.constructArrayType(class0);
      assertFalse(arrayType0.isInterface());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      Class<Long> class0 = Long.TYPE;
      TypeReference<MapLikeType> typeReference0 = (TypeReference<MapLikeType>) mock(TypeReference.class, new ViolatedAssumptionAnswer());
      doReturn(class0).when(typeReference0).getType();
      ObjectReader objectReader0 = objectMapper0.readerFor(typeReference0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<CollectionLikeType> class0 = CollectionLikeType.class;
      MapLikeType mapLikeType0 = typeFactory0.constructRawMapLikeType(class0);
      assertFalse(mapLikeType0.isFinal());
      assertFalse(mapLikeType0.isJavaLangObject());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      JavaType javaType0 = typeFactory0.constructFromCanonical("byte");
      assertTrue(javaType0.isConcrete());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      ClassLoader classLoader0 = ClassLoader.getSystemClassLoader();
      TypeFactory typeFactory1 = typeFactory0.withClassLoader(classLoader0);
      try { 
        typeFactory1.findClass("4K4Z_JDVZ9#MP");
        fail("Expecting exception: ClassNotFoundException");
      
      } catch(ClassNotFoundException e) {
         //
         // 4K4Z_JDVZ9#MP
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<MapType> class0 = MapType.class;
      PlaceholderForType placeholderForType0 = new PlaceholderForType((-1441610188));
      TypeBindings typeBindings0 = placeholderForType0._bindings;
      JavaType[] javaTypeArray0 = typeFactory0.findTypeParameters(class0, class0, typeBindings0);
      assertEquals(0, javaTypeArray0.length);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<String> class0 = String.class;
      PlaceholderForType placeholderForType0 = new PlaceholderForType(4115);
      JavaType javaType0 = placeholderForType0.getSuperClass();
      typeFactory0.constructReferenceType(class0, javaType0);
      assertTrue(javaType0.isJavaLangObject());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      TypeFactory typeFactory1 = typeFactory0.withCache((LRUMap<Object, JavaType>) null);
      assertFalse(typeFactory1.equals((Object)typeFactory0));
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<Properties> class0 = Properties.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      ArrayType arrayType0 = typeFactory0.constructArrayType((JavaType) simpleType0);
      assertFalse(arrayType0.isCollectionLikeType());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      LRUMap<Object, JavaType> lRUMap0 = new LRUMap<Object, JavaType>(3, 3);
      TypeFactory typeFactory0 = new TypeFactory(lRUMap0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      TypeFactory typeFactory0 = new TypeFactory((LRUMap<Object, JavaType>) null);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      TypeModifier typeModifier0 = mock(TypeModifier.class, new ViolatedAssumptionAnswer());
      TypeFactory typeFactory1 = typeFactory0.withModifier(typeModifier0);
      TypeModifier typeModifier1 = mock(TypeModifier.class, new ViolatedAssumptionAnswer());
      TypeFactory typeFactory2 = typeFactory1.withModifier(typeModifier1);
      assertFalse(typeFactory2.equals((Object)typeFactory0));
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      TypeFactory typeFactory1 = typeFactory0.withModifier((TypeModifier) null);
      assertFalse(typeFactory1.equals((Object)typeFactory0));
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      // Undeclared exception!
      try { 
        TypeFactory.rawClass((Type) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Unrecognized Type: [null]
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Class<Properties> class0 = Properties.class;
      Class<?> class1 = TypeFactory.rawClass(class0);
      assertFalse(class1.isEnum());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      try { 
        typeFactory0.findClass("K.(UMcD_+!14_");
        fail("Expecting exception: ClassNotFoundException");
      
      } catch(ClassNotFoundException e) {
         //
         // Class 'K/(UMcD_+!14_.class' should be in target project, but could not be found!
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<?> class0 = typeFactory0.findClass("int");
      assertFalse(class0.isAnnotation());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<?> class0 = typeFactory0.findClass("long");
      assertEquals("long", class0.toString());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<?> class0 = typeFactory0.findClass("float");
      assertEquals("float", class0.toString());
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<?> class0 = typeFactory0.findClass("double");
      assertEquals("double", class0.toString());
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<?> class0 = typeFactory0.findClass("boolean");
      assertEquals("boolean", class0.toString());
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<?> class0 = typeFactory0.findClass("char");
      assertEquals("char", class0.toString());
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<?> class0 = typeFactory0.findClass("short");
      assertEquals("short", class0.toString());
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<?> class0 = typeFactory0.findClass("void");
      assertEquals("void", class0.toString());
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<ArrayList> class0 = ArrayList.class;
      CollectionType collectionType0 = typeFactory0.constructCollectionType(class0, class0);
      JavaType javaType0 = typeFactory0.constructSpecializedType(collectionType0, class0);
      assertEquals(1, javaType0.containedTypeCount());
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<Integer> class0 = Integer.class;
      JavaType javaType0 = TypeFactory.unknownType();
      JavaType javaType1 = typeFactory0.constructSpecializedType(javaType0, class0);
      assertFalse(javaType1.isJavaLangObject());
      assertTrue(javaType0.isJavaLangObject());
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<MapLikeType> class0 = MapLikeType.class;
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      JavaType javaType0 = typeFactory0._fromAny((ClassStack) null, class0, typeBindings0);
      JavaType[] javaTypeArray0 = new JavaType[6];
      javaTypeArray0[1] = javaType0;
      TreeMap<Integer, Object> treeMap0 = new TreeMap<Integer, Object>();
      PlaceholderForType placeholderForType0 = new PlaceholderForType(46);
      ReferenceType referenceType0 = new ReferenceType(class0, typeBindings0, javaType0, javaTypeArray0, javaTypeArray0[1], javaTypeArray0[3], treeMap0, placeholderForType0, true);
      Class<MapType> class1 = MapType.class;
      JavaType javaType1 = typeFactory0.constructSpecializedType(referenceType0, class1);
      assertFalse(javaType1.equals((Object)javaType0));
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<TreeSet> class0 = TreeSet.class;
      CollectionType collectionType0 = typeFactory0.constructCollectionType(class0, class0);
      JavaType javaType0 = typeFactory0.constructGeneralizedType(collectionType0, class0);
      assertEquals(1, javaType0.containedTypeCount());
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<TreeSet> class0 = TreeSet.class;
      CollectionType collectionType0 = typeFactory0.constructCollectionType(class0, class0);
      Class<Object> class1 = Object.class;
      typeFactory0.constructGeneralizedType(collectionType0, class1);
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<Properties> class0 = Properties.class;
      JavaType javaType0 = typeFactory0.uncheckedSimpleType(class0);
      Class<Object> class1 = Object.class;
      // Undeclared exception!
      try { 
        typeFactory0.constructGeneralizedType(javaType0, class1);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Internal error: class java.lang.Object not included as super-type for [simple type, class java.util.Properties]
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      TypeParser typeParser0 = new TypeParser(typeFactory0);
      JavaType javaType0 = typeParser0.parse("byte");
      PlaceholderForType placeholderForType0 = new PlaceholderForType(2556);
      JavaType javaType1 = typeFactory0.moreSpecificType(javaType0, placeholderForType0);
      assertTrue(javaType1.isPrimitive());
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      JavaType javaType0 = typeFactory0.moreSpecificType((JavaType) null, (JavaType) null);
      assertNull(javaType0);
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<Properties> class0 = Properties.class;
      Class<MapLikeType> class1 = MapLikeType.class;
      MapLikeType mapLikeType0 = typeFactory0.constructMapLikeType(class0, class1, class0);
      JavaType[] javaTypeArray0 = new JavaType[5];
      javaTypeArray0[0] = (JavaType) mapLikeType0;
      javaTypeArray0[3] = (JavaType) mapLikeType0;
      ReferenceType referenceType0 = ReferenceType.upgradeFrom(javaTypeArray0[3], javaTypeArray0[0]);
      JavaType javaType0 = typeFactory0.moreSpecificType(referenceType0, javaTypeArray0[1]);
      assertSame(javaType0, referenceType0);
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      PlaceholderForType placeholderForType0 = new PlaceholderForType((-1));
      JavaType javaType0 = typeFactory0.moreSpecificType(placeholderForType0, placeholderForType0);
      assertFalse(javaType0.isPrimitive());
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<HashSet> class0 = HashSet.class;
      CollectionType collectionType0 = typeFactory0.constructCollectionType(class0, class0);
      PlaceholderForType placeholderForType0 = new PlaceholderForType(0);
      JavaType javaType0 = typeFactory0.moreSpecificType(placeholderForType0, collectionType0);
      assertEquals(1, javaType0.containedTypeCount());
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      JavaType javaType0 = typeFactory0._unknownType();
      Class<CollectionType> class0 = CollectionType.class;
      JavaType javaType1 = typeFactory0.constructType((Type) javaType0, (Class<?>) class0);
      assertFalse(javaType1.isInterface());
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      // Undeclared exception!
      try { 
        typeFactory0.constructType((Type) null, (Class<?>) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Unrecognized Type: [null]
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<Properties> class0 = Properties.class;
      JavaType javaType0 = typeFactory0.constructType((Type) class0, (Class<?>) class0);
      assertTrue(javaType0.hasContentType());
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<HashSet> class0 = HashSet.class;
      CollectionType collectionType0 = typeFactory0.constructCollectionType(class0, class0);
      JavaType javaType0 = typeFactory0.constructType((Type) collectionType0, (JavaType) collectionType0);
      assertEquals(1, javaType0.containedTypeCount());
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<TreeSet> class0 = TreeSet.class;
      Class<Properties> class1 = Properties.class;
      Class<AnnotationIntrospector.ReferenceProperty.Type> class2 = AnnotationIntrospector.ReferenceProperty.Type.class;
      // Undeclared exception!
      try { 
        typeFactory0.constructMapType(class1, class0, class2);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Can not create TypeBindings for class java.util.Properties with 2 type parameters: class expects 0
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeBindings", e);
      }
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<Object> class0 = Object.class;
      Class<DeserializationFeature> class1 = DeserializationFeature.class;
      Class<Integer>[] classArray0 = (Class<Integer>[]) Array.newInstance(Class.class, 9);
      // Undeclared exception!
      try { 
        typeFactory0.constructParametrizedType(class0, class1, classArray0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      AtomicReference<String> atomicReference0 = new AtomicReference<String>("z4_Aae:43`: TyaIM");
      objectMapper0.writeValueAsString(atomicReference0);
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      assertNotNull(typeFactory0);
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<String> class0 = String.class;
      JavaType javaType0 = typeFactory0.uncheckedSimpleType(class0);
      assertFalse(javaType0.isJavaLangObject());
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<Properties> class0 = Properties.class;
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_INT;
      TypeModifier typeModifier0 = mock(TypeModifier.class, new ViolatedAssumptionAnswer());
      doReturn(simpleType0, (JavaType) null).when(typeModifier0).modifyType(any(com.fasterxml.jackson.databind.JavaType.class) , any(java.lang.reflect.Type.class) , any(com.fasterxml.jackson.databind.type.TypeBindings.class) , any(com.fasterxml.jackson.databind.type.TypeFactory.class));
      doReturn((String) null).when(typeModifier0).toString();
      TypeFactory typeFactory1 = typeFactory0.withModifier(typeModifier0);
      Class<TreeSet> class1 = TreeSet.class;
      // Undeclared exception!
      try { 
        typeFactory1.constructCollectionType(class1, class0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // TypeModifier null (of type com.fasterxml.jackson.databind.type.TypeModifier$MockitoMock$1861423632) return null for type [simple type, class int]
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test55()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      TypeModifier typeModifier0 = mock(TypeModifier.class, new ViolatedAssumptionAnswer());
      doReturn((JavaType) null).when(typeModifier0).modifyType(any(com.fasterxml.jackson.databind.JavaType.class) , any(java.lang.reflect.Type.class) , any(com.fasterxml.jackson.databind.type.TypeBindings.class) , any(com.fasterxml.jackson.databind.type.TypeFactory.class));
      doReturn((String) null).when(typeModifier0).toString();
      TypeFactory typeFactory1 = typeFactory0.withModifier(typeModifier0);
      Class<TreeSet> class0 = TreeSet.class;
      // Undeclared exception!
      try { 
        typeFactory1.constructCollectionType(class0, class0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // TypeModifier Mock for TypeModifier, hashCode: 196876098 (of type com.fasterxml.jackson.databind.type.TypeModifier$MockitoMock$1861423632) return null for type [simple type, class java.lang.Object]
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test56()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<ArrayNode> class0 = ArrayNode.class;
      Class<JsonSerializer> class1 = JsonSerializer.class;
      MapLikeType mapLikeType0 = typeFactory0.constructMapLikeType(class0, class1, class0);
      assertTrue(mapLikeType0.isConcrete());
  }

  @Test(timeout = 4000)
  public void test57()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<Integer> class0 = Integer.class;
      Class<SerializationFeature> class1 = SerializationFeature.class;
      JavaType[] javaTypeArray0 = new JavaType[4];
      PlaceholderForType placeholderForType0 = new PlaceholderForType(2616);
      TypeBindings typeBindings0 = placeholderForType0.getBindings();
      CollectionType collectionType0 = new CollectionType(placeholderForType0, placeholderForType0);
      AtomicReference<CollectionLikeType> atomicReference0 = new AtomicReference<CollectionLikeType>(collectionType0);
      ReferenceType referenceType0 = new ReferenceType(class0, typeBindings0, placeholderForType0, javaTypeArray0, placeholderForType0, placeholderForType0, class1, atomicReference0, false);
      Class<ArrayList> class2 = ArrayList.class;
      CollectionType collectionType1 = typeFactory0.constructCollectionType((Class<? extends Collection>) class2, (JavaType) referenceType0);
      assertTrue(collectionType1.hasHandlers());
  }

  @Test(timeout = 4000)
  public void test58()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      Class<Object> class0 = Object.class;
      objectMapper0.readerFor(class0);
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      TypeParser typeParser0 = new TypeParser(typeFactory0);
      typeParser0.parse("byte");
      Class<ArrayList> class1 = ArrayList.class;
      Class<Integer> class2 = Integer.class;
      typeFactory0.constructCollectionType(class1, class2);
      PlaceholderForType placeholderForType0 = new PlaceholderForType(2556);
      TypeBindings typeBindings0 = placeholderForType0._bindings;
      Class<Object> class3 = Object.class;
      ClassStack classStack0 = new ClassStack(class3);
      Class<PlaceholderForType> class4 = PlaceholderForType.class;
      JavaType[] javaTypeArray0 = typeFactory0._resolveSuperInterfaces(classStack0, class4, typeBindings0);
      assertEquals(1, javaTypeArray0.length);
  }

  @Test(timeout = 4000)
  public void test59()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      Class<Object> class0 = Object.class;
      ObjectReader objectReader0 = objectMapper0.readerFor(class0);
      TypeFactory typeFactory0 = objectReader0.getTypeFactory();
      TypeBindings typeBindings0 = TypeFactory.EMPTY_BINDINGS;
      ParameterizedType parameterizedType0 = mock(ParameterizedType.class, new ViolatedAssumptionAnswer());
      doReturn((Type[]) null).when(parameterizedType0).getActualTypeArguments();
      doReturn((Type) null).when(parameterizedType0).getRawType();
      // Undeclared exception!
      try { 
        typeFactory0._fromParamType((ClassStack) null, parameterizedType0, typeBindings0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }
}