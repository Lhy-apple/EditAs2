/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 15:02:03 GMT 2023
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
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.databind.node.LongNode;
import com.fasterxml.jackson.databind.type.ArrayType;
import com.fasterxml.jackson.databind.type.ClassStack;
import com.fasterxml.jackson.databind.type.CollectionLikeType;
import com.fasterxml.jackson.databind.type.CollectionType;
import com.fasterxml.jackson.databind.type.MapLikeType;
import com.fasterxml.jackson.databind.type.MapType;
import com.fasterxml.jackson.databind.type.ReferenceType;
import com.fasterxml.jackson.databind.type.ResolvedRecursiveType;
import com.fasterxml.jackson.databind.type.SimpleType;
import com.fasterxml.jackson.databind.type.TypeBindings;
import com.fasterxml.jackson.databind.type.TypeFactory;
import com.fasterxml.jackson.databind.type.TypeModifier;
import com.fasterxml.jackson.databind.type.TypeParser;
import java.lang.reflect.Array;
import java.lang.reflect.GenericArrayType;
import java.lang.reflect.ParameterizedType;
import java.lang.reflect.Type;
import java.time.chrono.ChronoLocalDate;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class TypeFactory_ESTest extends TypeFactory_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      JavaType[] javaTypeArray0 = new JavaType[6];
      Class<Properties> class0 = Properties.class;
      Class<String> class1 = String.class;
      // Undeclared exception!
      try { 
        typeFactory0.constructSimpleType(class0, class1, javaTypeArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Can not create TypeBindings for class java.util.Properties with 6 type parameters: class expects 0
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeBindings", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<HashSet> class0 = HashSet.class;
      CollectionType collectionType0 = typeFactory0.constructRawCollectionType(class0);
      assertFalse(collectionType0.isArrayType());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      TypeParser typeParser0 = new TypeParser(typeFactory0);
      TypeFactory typeFactory1 = new TypeFactory(typeParser0, (TypeModifier[]) null);
      assertFalse(typeFactory1.equals((Object)typeFactory0));
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      ObjectMapper objectMapper0 = new ObjectMapper();
      objectMapper0.readerForUpdating(javaType0);
      assertTrue(javaType0.isJavaLangObject());
      assertTrue(javaType0.isConcrete());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<DeserializationFeature> class0 = DeserializationFeature.class;
      JavaType[] javaTypeArray0 = new JavaType[0];
      JavaType javaType0 = typeFactory0.constructParametrizedType(class0, class0, javaTypeArray0);
      JavaType javaType1 = typeFactory0.constructSpecializedType(javaType0, class0);
      assertFalse(javaType1.isInterface());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      ObjectMapper objectMapper0 = new ObjectMapper();
      objectMapper0.readerFor(javaType0);
      assertFalse(javaType0.isFinal());
      assertTrue(javaType0.isJavaLangObject());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<LongNode> class0 = LongNode.class;
      Class<MapType>[] classArray0 = (Class<MapType>[]) Array.newInstance(Class.class, 0);
      JavaType javaType0 = typeFactory0.constructParametrizedType(class0, class0, classArray0);
      JavaType javaType1 = typeFactory0.constructReferenceType(class0, javaType0);
      assertTrue(javaType1.isConcrete());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<ReferenceType> class0 = ReferenceType.class;
      ClassStack classStack0 = new ClassStack(class0);
      ArrayList<JavaType> arrayList0 = new ArrayList<JavaType>();
      TypeBindings typeBindings0 = TypeBindings.create((Class<?>) class0, (List<JavaType>) arrayList0);
      // Undeclared exception!
      try { 
        typeFactory0._fromArrayType(classStack0, (GenericArrayType) null, typeBindings0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      typeFactory0.clearCache();
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<SerializationFeature> class0 = SerializationFeature.class;
      CollectionLikeType collectionLikeType0 = typeFactory0.constructRawCollectionLikeType(class0);
      assertFalse(collectionLikeType0.isJavaLangObject());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<LinkedList> class0 = LinkedList.class;
      CollectionType collectionType0 = typeFactory0.constructCollectionType(class0, class0);
      Class<CollectionType> class1 = CollectionType.class;
      // Undeclared exception!
      try { 
        typeFactory0.constructGeneralizedType(collectionType0, class1);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Class com.fasterxml.jackson.databind.type.CollectionType not a super-type of [collection type; class java.util.LinkedList, contains [collection type; class java.util.LinkedList, contains [simple type, class java.lang.Object]]]
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<MapType> class0 = MapType.class;
      CollectionLikeType collectionLikeType0 = typeFactory0.constructCollectionLikeType(class0, class0);
      Class<?> class1 = TypeFactory.rawClass(collectionLikeType0);
      assertEquals("class com.fasterxml.jackson.databind.type.MapType", class1.toString());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      // Undeclared exception!
      try { 
        objectMapper0.readerFor((TypeReference<?>) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<Properties> class0 = Properties.class;
      // Undeclared exception!
      try { 
        typeFactory0.constructRawMapType(class0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Can not create TypeBindings for class java.util.Properties with 2 type parameters: class expects 0
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeBindings", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<Properties> class0 = Properties.class;
      MapLikeType mapLikeType0 = typeFactory0.constructRawMapLikeType(class0);
      assertFalse(mapLikeType0.isEnumType());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      ClassLoader classLoader0 = ClassLoader.getSystemClassLoader();
      TypeFactory typeFactory1 = typeFactory0.withClassLoader(classLoader0);
      try { 
        typeFactory1.findClass("%zm5r!9M2uLLjc");
        fail("Expecting exception: ClassNotFoundException");
      
      } catch(ClassNotFoundException e) {
         //
         // %zm5r!9M2uLLjc
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<MapLikeType> class0 = MapLikeType.class;
      TypeBindings typeBindings0 = TypeFactory.EMPTY_BINDINGS;
      JavaType[] javaTypeArray0 = typeFactory0.findTypeParameters(class0, class0, typeBindings0);
      assertEquals(0, javaTypeArray0.length);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_COMPARABLE;
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      ArrayType arrayType0 = typeFactory0.constructArrayType((JavaType) simpleType0);
      assertTrue(arrayType0.isConcrete());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      TypeFactory typeFactory1 = typeFactory0.withModifier((TypeModifier) null);
      assertNotSame(typeFactory0, typeFactory1);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      TypeModifier typeModifier0 = mock(TypeModifier.class, new ViolatedAssumptionAnswer());
      TypeFactory typeFactory1 = typeFactory0.withModifier(typeModifier0);
      TypeModifier typeModifier1 = mock(TypeModifier.class, new ViolatedAssumptionAnswer());
      TypeFactory typeFactory2 = typeFactory1.withModifier(typeModifier1);
      assertNotSame(typeFactory2, typeFactory0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Class<Properties> class0 = Properties.class;
      Class<?> class1 = TypeFactory.rawClass(class0);
      assertFalse(class1.isAnnotation());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      try { 
        typeFactory0.findClass("=v.{A/s-5=k-W");
        fail("Expecting exception: ClassNotFoundException");
      
      } catch(ClassNotFoundException e) {
         //
         // =v.{A/s-5=k-W
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<?> class0 = typeFactory0.findClass("long");
      assertEquals("long", class0.toString());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<?> class0 = typeFactory0.findClass("int");
      assertFalse(class0.isEnum());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<?> class0 = typeFactory0.findClass("float");
      assertEquals("float", class0.toString());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<?> class0 = typeFactory0.findClass("double");
      assertEquals("double", class0.toString());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<?> class0 = typeFactory0.findClass("boolean");
      assertEquals("boolean", class0.toString());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<?> class0 = typeFactory0.findClass("byte");
      assertEquals("byte", class0.toString());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      JavaType javaType0 = typeFactory0.constructFromCanonical("char");
      assertFalse(javaType0.isMapLikeType());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<?> class0 = typeFactory0.findClass("short");
      assertEquals("short", class0.toString());
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<?> class0 = typeFactory0.findClass("void");
      assertEquals("void", class0.toString());
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      JavaType javaType0 = TypeFactory.unknownType();
      assertTrue(javaType0.isJavaLangObject());
      
      Class<LongNode> class0 = LongNode.class;
      JavaType javaType1 = typeFactory0.constructSpecializedType(javaType0, class0);
      assertFalse(javaType1.isJavaLangObject());
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_INT;
      Class<CollectionLikeType> class0 = CollectionLikeType.class;
      // Undeclared exception!
      try { 
        typeFactory0.constructSpecializedType(simpleType0, class0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Class com.fasterxml.jackson.databind.type.CollectionLikeType not subtype of [simple type, class int]
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<DeserializationFeature> class0 = DeserializationFeature.class;
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_COMPARABLE;
      JavaType javaType0 = typeFactory0.constructSpecializedType(simpleType0, class0);
      assertFalse(javaType0.isInterface());
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<MapLikeType> class0 = MapLikeType.class;
      Class<MapType> class1 = MapType.class;
      MapLikeType mapLikeType0 = typeFactory0.constructMapLikeType(class0, class0, class1);
      Class<Object> class2 = Object.class;
      typeFactory0.constructGeneralizedType(mapLikeType0, class2);
      assertFalse(mapLikeType0.isFinal());
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<LinkedList> class0 = LinkedList.class;
      CollectionType collectionType0 = typeFactory0.constructCollectionType(class0, class0);
      JavaType javaType0 = typeFactory0.constructGeneralizedType(collectionType0, class0);
      assertEquals(1, javaType0.containedTypeCount());
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<CollectionLikeType> class0 = CollectionLikeType.class;
      JavaType javaType0 = typeFactory0.uncheckedSimpleType(class0);
      Class<Object> class1 = Object.class;
      // Undeclared exception!
      try { 
        typeFactory0.constructGeneralizedType(javaType0, class1);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Internal error: class java.lang.Object not included as super-type for [simple type, class com.fasterxml.jackson.databind.type.CollectionLikeType]
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<AnnotationIntrospector.ReferenceProperty.Type> class0 = AnnotationIntrospector.ReferenceProperty.Type.class;
      Class<ResolvedRecursiveType> class1 = ResolvedRecursiveType.class;
      JavaType[] javaTypeArray0 = typeFactory0.findTypeParameters(class0, class1);
      assertEquals(0, javaTypeArray0.length);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      JavaType javaType0 = typeFactory0._unknownType();
      Class<MapLikeType> class0 = MapLikeType.class;
      Class<MapType> class1 = MapType.class;
      MapLikeType mapLikeType0 = typeFactory0.constructMapLikeType(class0, class0, class1);
      JavaType javaType1 = typeFactory0.moreSpecificType(mapLikeType0, javaType0);
      assertTrue(javaType0.isJavaLangObject());
      assertFalse(javaType1.isFinal());
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      JavaType javaType0 = typeFactory0.moreSpecificType((JavaType) null, (JavaType) null);
      assertNull(javaType0);
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_CLASS;
      JavaType javaType0 = typeFactory0.moreSpecificType(simpleType0, (JavaType) null);
      assertFalse(javaType0.isContainerType());
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<SerializationFeature> class0 = SerializationFeature.class;
      ArrayType arrayType0 = typeFactory0.constructArrayType(class0);
      JavaType javaType0 = typeFactory0.moreSpecificType(arrayType0, arrayType0);
      assertTrue(javaType0.isArrayType());
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<LinkedList> class0 = LinkedList.class;
      JavaType javaType0 = typeFactory0.constructType((Type) class0, (Class<?>) class0);
      JavaType javaType1 = typeFactory0._unknownType();
      typeFactory0.moreSpecificType(javaType1, javaType0);
      assertTrue(javaType1.isJavaLangObject());
      assertFalse(javaType1.isFinal());
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
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
  public void test44()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<ReferenceType> class0 = ReferenceType.class;
      ArrayType arrayType0 = typeFactory0.constructArrayType(class0);
      JavaType javaType0 = typeFactory0.constructType((Type) arrayType0, (JavaType) arrayType0);
      assertFalse(javaType0.isPrimitive());
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<SerializationFeature> class0 = SerializationFeature.class;
      ArrayType arrayType0 = typeFactory0.constructArrayType(class0);
      JavaType javaType0 = typeFactory0.constructType((Type) arrayType0, (JavaType) null);
      assertTrue(javaType0.isArrayType());
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<LinkedList> class0 = LinkedList.class;
      CollectionType collectionType0 = typeFactory0.constructCollectionType(class0, class0);
      CollectionLikeType collectionLikeType0 = typeFactory0.constructCollectionLikeType((Class<?>) class0, (JavaType) collectionType0);
      assertFalse(collectionLikeType0.equals((Object)collectionType0));
      assertNotSame(collectionLikeType0, collectionType0);
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<ReferenceType> class0 = ReferenceType.class;
      Class<DeserializationFeature> class1 = DeserializationFeature.class;
      // Undeclared exception!
      try { 
        typeFactory0.constructMapType((Class<? extends Map>) null, class0, class1);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeBindings$TypeParamStash", e);
      }
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<Properties> class0 = Properties.class;
      // Undeclared exception!
      try { 
        typeFactory0.constructMapType(class0, class0, class0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Can not create TypeBindings for class java.util.Properties with 2 type parameters: class expects 0
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeBindings", e);
      }
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<ChronoLocalDate> class0 = ChronoLocalDate.class;
      Class<ArrayType>[] classArray0 = (Class<ArrayType>[]) Array.newInstance(Class.class, 1);
      // Undeclared exception!
      try { 
        typeFactory0.constructParametricType(class0, classArray0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<LinkedList> class0 = LinkedList.class;
      CollectionType collectionType0 = typeFactory0.constructCollectionType(class0, class0);
      TypeBindings typeBindings0 = TypeBindings.createIfNeeded((Class<?>) class0, (JavaType) collectionType0);
      JavaType[] javaTypeArray0 = new JavaType[5];
      JavaType javaType0 = typeFactory0._constructSimple(class0, typeBindings0, collectionType0, javaTypeArray0);
      assertEquals(1, collectionType0.containedTypeCount());
      assertEquals(1, javaType0.containedTypeCount());
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<String> class0 = String.class;
      JavaType javaType0 = typeFactory0.uncheckedSimpleType(class0);
      assertFalse(javaType0.isJavaLangObject());
      assertNotNull(javaType0);
      assertFalse(javaType0.isAbstract());
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<Long> class0 = Long.TYPE;
      JavaType javaType0 = typeFactory0.constructType((Type) class0, (Class<?>) class0);
      assertTrue(javaType0.isPrimitive());
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_CLASS;
      TypeModifier typeModifier0 = mock(TypeModifier.class, new ViolatedAssumptionAnswer());
      doReturn(simpleType0).when(typeModifier0).modifyType(any(com.fasterxml.jackson.databind.JavaType.class) , any(java.lang.reflect.Type.class) , any(com.fasterxml.jackson.databind.type.TypeBindings.class) , any(com.fasterxml.jackson.databind.type.TypeFactory.class));
      TypeFactory typeFactory1 = typeFactory0.withModifier(typeModifier0);
      Class<ObjectMapper.DefaultTyping> class0 = ObjectMapper.DefaultTyping.class;
      MapLikeType mapLikeType0 = typeFactory1.constructRawMapLikeType(class0);
      assertFalse(mapLikeType0.isJavaLangObject());
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      TypeModifier typeModifier0 = mock(TypeModifier.class, new ViolatedAssumptionAnswer());
      doReturn((JavaType) null).when(typeModifier0).modifyType(any(com.fasterxml.jackson.databind.JavaType.class) , any(java.lang.reflect.Type.class) , any(com.fasterxml.jackson.databind.type.TypeBindings.class) , any(com.fasterxml.jackson.databind.type.TypeFactory.class));
      doReturn((String) null).when(typeModifier0).toString();
      TypeFactory typeFactory1 = typeFactory0.withModifier(typeModifier0);
      Class<DeserializationFeature> class0 = DeserializationFeature.class;
      // Undeclared exception!
      try { 
        typeFactory1.constructRawMapLikeType(class0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // TypeModifier null (of type com.fasterxml.jackson.databind.type.TypeModifier$MockitoMock$516072789) return null for type [simple type, class java.lang.Enum]
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test55()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<String> class0 = String.class;
      ArrayType arrayType0 = typeFactory0.constructArrayType(class0);
      ObjectReader objectReader0 = objectMapper0.readerFor((JavaType) arrayType0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test56()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<Integer> class0 = Integer.class;
      MapLikeType mapLikeType0 = typeFactory0.constructRawMapLikeType(class0);
      assertTrue(mapLikeType0.isFinal());
  }

  @Test(timeout = 4000)
  public void test57()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      ParameterizedType parameterizedType0 = mock(ParameterizedType.class, new ViolatedAssumptionAnswer());
      doReturn((Type[]) null).when(parameterizedType0).getActualTypeArguments();
      doReturn((Type) null).when(parameterizedType0).getRawType();
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
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