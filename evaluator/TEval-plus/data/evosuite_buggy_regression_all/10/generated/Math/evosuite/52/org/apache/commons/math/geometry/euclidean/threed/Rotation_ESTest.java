/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 14:26:03 GMT 2023
 */

package org.apache.commons.math.geometry.euclidean.threed;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.apache.commons.math.geometry.euclidean.threed.Rotation;
import org.apache.commons.math.geometry.euclidean.threed.RotationOrder;
import org.apache.commons.math.geometry.euclidean.threed.Vector3D;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Rotation_ESTest extends Rotation_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      double[][] doubleArray0 = new double[3][3];
      Rotation rotation0 = new Rotation(doubleArray0, 2.7922362104828187);
      Rotation.distance(rotation0, rotation0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Rotation rotation0 = Rotation.IDENTITY;
      Rotation rotation1 = rotation0.revert();
      assertEquals(0.0, rotation1.getQ1(), 0.01);
      assertEquals((-1.0), rotation1.getQ0(), 0.01);
      assertEquals(0.0, rotation1.getQ2(), 0.01);
      assertEquals(0.0, rotation1.getQ3(), 0.01);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Rotation rotation0 = new Rotation(1.6668541182173628, 1.6668541182173628, (-1314.2506), 1.6668541182173628, true);
      double double0 = rotation0.getQ3();
      assertEquals(0.0012682893934880125, double0, 0.01);
      assertEquals((-0.9999975871601106), rotation0.getQ2(), 0.01);
      assertEquals(0.0012682893934880125, rotation0.getQ0(), 0.01);
      assertEquals(0.0012682893934880125, rotation0.getQ1(), 0.01);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Rotation rotation0 = Rotation.IDENTITY;
      double double0 = rotation0.getQ1();
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      RotationOrder rotationOrder0 = RotationOrder.ZYX;
      Rotation rotation0 = new Rotation(rotationOrder0, (-534.46), (-534.46), (-3210.41697));
      double double0 = rotation0.getQ2();
      assertEquals((-0.21523042067064388), double0, 0.01);
      assertEquals(0.10242347570747462, rotation0.getQ1(), 0.01);
      assertEquals((-0.9578631090538556), rotation0.getQ0(), 0.01);
      assertEquals((-0.16026091836633038), rotation0.getQ3(), 0.01);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Vector3D vector3D0 = Vector3D.MINUS_K;
      Rotation rotation0 = new Rotation(vector3D0, vector3D0);
      double double0 = rotation0.getQ0();
      assertEquals(0.0, rotation0.getQ2(), 0.01);
      assertEquals(0.0, rotation0.getQ1(), 0.01);
      assertEquals(1.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Rotation rotation0 = Rotation.IDENTITY;
      RotationOrder rotationOrder0 = RotationOrder.XYZ;
      double[] doubleArray0 = rotation0.getAngles(rotationOrder0);
      double[][] doubleArray1 = new double[3][0];
      doubleArray1[0] = doubleArray0;
      doubleArray1[1] = doubleArray0;
      Rotation rotation1 = null;
      try {
        rotation1 = new Rotation(doubleArray1, 0.0);
        fail("Expecting exception: Exception");
      
      } catch(Throwable e) {
         //
         // a 3x3 matrix cannot be a rotation matrix
         //
         verifyException("org.apache.commons.math.geometry.euclidean.threed.Rotation", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Rotation rotation0 = new Rotation(1.6668541182173628, 1.6668541182173628, (-1314.2506), 1.6668541182173628, true);
      double[][] doubleArray0 = rotation0.getMatrix();
      Rotation rotation1 = new Rotation(doubleArray0, 1.6668541182173628);
      assertEquals((-0.0012682893934880127), rotation1.getQ0(), 0.01);
      assertEquals((-0.9999975871601106), rotation0.getQ2(), 0.01);
      assertEquals(0.0012682893934880125, rotation0.getQ3(), 0.01);
      assertEquals((-0.001268289393488013), rotation1.getQ3(), 0.01);
      assertEquals(0.0012682893934880125, rotation0.getQ1(), 0.01);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Vector3D vector3D0 = Vector3D.ZERO;
      Rotation rotation0 = null;
      try {
        rotation0 = new Rotation(vector3D0, 1403.736);
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // zero norm for rotation axis
         //
         verifyException("org.apache.commons.math.MathRuntimeException", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      double[][] doubleArray0 = new double[0][8];
      Rotation rotation0 = null;
      try {
        rotation0 = new Rotation(doubleArray0, 1.304E19);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 0
         //
         verifyException("org.apache.commons.math.geometry.euclidean.threed.Rotation", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      double[][] doubleArray0 = new double[3][8];
      Rotation rotation0 = null;
      try {
        rotation0 = new Rotation(doubleArray0, 1.304E19);
        fail("Expecting exception: Exception");
      
      } catch(Throwable e) {
         //
         // a 3x8 matrix cannot be a rotation matrix
         //
         verifyException("org.apache.commons.math.geometry.euclidean.threed.Rotation", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      double[][] doubleArray0 = new double[3][5];
      double[] doubleArray1 = new double[3];
      doubleArray0[0] = doubleArray1;
      Rotation rotation0 = null;
      try {
        rotation0 = new Rotation(doubleArray0, (-226.9824970491));
        fail("Expecting exception: Exception");
      
      } catch(Throwable e) {
         //
         // a 3x3 matrix cannot be a rotation matrix
         //
         verifyException("org.apache.commons.math.geometry.euclidean.threed.Rotation", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Rotation rotation0 = new Rotation(270.43079926, (-1186.83242663), (-1186.83242663), 270.43079926, true);
      double[][] doubleArray0 = rotation0.getMatrix();
      Rotation rotation1 = new Rotation(doubleArray0, 1.2599210498948732);
      assertEquals((-0.6894355527255781), rotation0.getQ2(), 0.01);
      assertEquals(0.6894355527255782, rotation1.getQ2(), 0.01);
      assertEquals((-0.6894355527255781), rotation0.getQ1(), 0.01);
      assertEquals(0.15709429855337348, rotation0.getQ0(), 0.01);
      assertEquals(0.15709429855337348, rotation0.getQ3(), 0.01);
      assertEquals((-0.15709429855337348), rotation1.getQ0(), 0.01);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Rotation rotation0 = new Rotation(0.5, (-638.06220555), (-638.06220555), (-3019.4380628012277), true);
      double[][] doubleArray0 = rotation0.getMatrix();
      Rotation rotation1 = new Rotation(doubleArray0, 0.5);
      assertEquals(1.586601924090219E-4, rotation0.getQ0(), 0.01);
      assertEquals((-1.5866019240901762E-4), rotation1.getQ0(), 0.01);
      assertEquals((-0.20247014460297577), rotation0.getQ2(), 0.01);
      assertEquals((-0.20247014460297577), rotation0.getQ1(), 0.01);
      assertEquals(0.20247014460297577, rotation1.getQ1(), 0.01);
      assertEquals((-0.9581292480223343), rotation0.getQ3(), 0.01);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Vector3D vector3D0 = Vector3D.ZERO;
      Rotation rotation0 = null;
      try {
        rotation0 = new Rotation(vector3D0, vector3D0, vector3D0, vector3D0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // zero norm for rotation defining vector
         //
         verifyException("org.apache.commons.math.MathRuntimeException", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Vector3D vector3D0 = Vector3D.NaN;
      Vector3D vector3D1 = Vector3D.ZERO;
      Rotation rotation0 = null;
      try {
        rotation0 = new Rotation(vector3D0, vector3D1, vector3D0, vector3D0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // zero norm for rotation defining vector
         //
         verifyException("org.apache.commons.math.MathRuntimeException", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Vector3D vector3D0 = Vector3D.ZERO;
      Vector3D vector3D1 = Vector3D.PLUS_J;
      Rotation rotation0 = null;
      try {
        rotation0 = new Rotation(vector3D1, vector3D1, vector3D0, vector3D1);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // zero norm for rotation defining vector
         //
         verifyException("org.apache.commons.math.MathRuntimeException", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Vector3D vector3D0 = Vector3D.MINUS_J;
      Vector3D vector3D1 = Vector3D.ZERO;
      Rotation rotation0 = null;
      try {
        rotation0 = new Rotation(vector3D0, vector3D0, vector3D0, vector3D1);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // zero norm for rotation defining vector
         //
         verifyException("org.apache.commons.math.MathRuntimeException", e);
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      RotationOrder rotationOrder0 = RotationOrder.ZYZ;
      Vector3D vector3D0 = rotationOrder0.getA2();
      Vector3D vector3D1 = Vector3D.MINUS_I;
      Rotation rotation0 = new Rotation(vector3D0, vector3D1, vector3D0, vector3D1);
      assertEquals(1.0, rotation0.getQ0(), 0.01);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Vector3D vector3D0 = Vector3D.MINUS_K;
      RotationOrder rotationOrder0 = RotationOrder.XYZ;
      Vector3D vector3D1 = new Vector3D(1.225743062930824E-8, (-6.930486595474513));
      Vector3D vector3D2 = rotationOrder0.getA3();
      Rotation rotation0 = new Rotation(vector3D0, vector3D1, vector3D2, vector3D1);
      RotationOrder rotationOrder1 = RotationOrder.ZXZ;
      try { 
        rotation0.getAngles(rotationOrder1);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Euler angles singularity
         //
         verifyException("org.apache.commons.math.geometry.euclidean.threed.Rotation", e);
      }
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Vector3D vector3D0 = new Vector3D((-0.33333333333333287), 0.7385090212904348, (-0.33333333333333287));
      Vector3D vector3D1 = new Vector3D(2597.501780637, vector3D0, 1160.757785478, vector3D0);
      Vector3D vector3D2 = new Vector3D(2597.501780637, 2597.501780637);
      Rotation rotation0 = new Rotation(vector3D1, vector3D2, vector3D0, vector3D0);
      assertEquals(4.4297268002233886E-10, rotation0.getQ0(), 0.01);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Vector3D vector3D0 = Vector3D.ZERO;
      Rotation rotation0 = null;
      try {
        rotation0 = new Rotation(vector3D0, vector3D0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // zero norm for rotation defining vector
         //
         verifyException("org.apache.commons.math.MathRuntimeException", e);
      }
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Vector3D vector3D0 = Vector3D.NaN;
      Rotation rotation0 = new Rotation(vector3D0, vector3D0, vector3D0, vector3D0);
      Vector3D vector3D1 = rotation0.getAxis();
      assertEquals(Double.NaN, vector3D1.getNormInf(), 0.01);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Vector3D vector3D0 = Vector3D.NEGATIVE_INFINITY;
      Rotation rotation0 = new Rotation(vector3D0, 757.65127469312);
      Vector3D vector3D1 = rotation0.getAxis();
      assertEquals((-0.26067623500942444), rotation0.getQ0(), 0.01);
      assertEquals(Double.NaN, rotation0.getQ3(), 0.01);
      assertEquals(Double.NaN, rotation0.getQ1(), 0.01);
      assertEquals(Double.NaN, rotation0.getQ2(), 0.01);
      assertEquals(Double.NaN, vector3D1.getX(), 0.01);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      double[][] doubleArray0 = new double[3][3];
      Rotation rotation0 = new Rotation(doubleArray0, 2.7922362104828187);
      RotationOrder rotationOrder0 = RotationOrder.ZXZ;
      Rotation rotation1 = new Rotation(rotationOrder0, (-920.897056), (-920.897056), (-920.897056));
      double double0 = Rotation.distance(rotation1, rotation0);
      assertEquals(2.9544739060217924, double0, 0.01);
      assertEquals(0.979019385689468, rotation1.getQ1(), 0.01);
      assertEquals(0.18684588083856038, rotation1.getQ0(), 0.01);
      assertEquals(0.08129981093384683, rotation1.getQ3(), 0.01);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Vector3D vector3D0 = Vector3D.NEGATIVE_INFINITY;
      Rotation rotation0 = new Rotation(vector3D0, 757.65127469312);
      double double0 = rotation0.IDENTITY.getAngle();
      assertEquals((-0.26067623500942444), rotation0.getQ0(), 0.01);
      assertEquals(Double.NaN, rotation0.getQ3(), 0.01);
      assertEquals(0.0, double0, 0.01);
      assertEquals(Double.NaN, rotation0.getQ1(), 0.01);
      assertEquals(Double.NaN, rotation0.getQ2(), 0.01);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Vector3D vector3D0 = Vector3D.NaN;
      Rotation rotation0 = new Rotation(vector3D0, vector3D0);
      double double0 = rotation0.getAngle();
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Rotation rotation0 = new Rotation(1.6668541182173628, 1.6668541182173628, (-1314.2506), 1.6668541182173628, true);
      RotationOrder rotationOrder0 = RotationOrder.YZX;
      double[] doubleArray0 = rotation0.getAngles(rotationOrder0);
      assertEquals((-0.9999975871601106), rotation0.getQ2(), 0.01);
      assertArrayEquals(new double[] {3.139059287158528, (-0.0025397925130960096), 0.0025333664312652423}, doubleArray0, 0.01);
      assertEquals(0.0012682893934880125, rotation0.getQ1(), 0.01);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Vector3D vector3D0 = Vector3D.MINUS_K;
      Rotation rotation0 = new Rotation(vector3D0, vector3D0);
      RotationOrder rotationOrder0 = RotationOrder.XYZ;
      Vector3D vector3D1 = rotation0.getAxis();
      Rotation rotation1 = new Rotation(vector3D0, vector3D1);
      try { 
        rotation1.getAngles(rotationOrder0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Cardan angles singularity
         //
         verifyException("org.apache.commons.math.geometry.euclidean.threed.Rotation", e);
      }
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      RotationOrder rotationOrder0 = RotationOrder.XYZ;
      Rotation rotation0 = new Rotation(1403.736, 2071.63, 1403.736, 2071.63, false);
      try { 
        rotation0.getAngles(rotationOrder0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Cardan angles singularity
         //
         verifyException("org.apache.commons.math.geometry.euclidean.threed.Rotation", e);
      }
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      RotationOrder rotationOrder0 = RotationOrder.XZY;
      Rotation rotation0 = new Rotation(1.6077569201477233, (-0.9999999999), (-833.7), (-833.7), false);
      try { 
        rotation0.getAngles(rotationOrder0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Cardan angles singularity
         //
         verifyException("org.apache.commons.math.geometry.euclidean.threed.Rotation", e);
      }
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      RotationOrder rotationOrder0 = RotationOrder.XZY;
      Rotation rotation0 = new Rotation(0.5, 0.5, 0.5, 0.5, false);
      try { 
        rotation0.getAngles(rotationOrder0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Cardan angles singularity
         //
         verifyException("org.apache.commons.math.geometry.euclidean.threed.Rotation", e);
      }
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      RotationOrder rotationOrder0 = RotationOrder.XZY;
      Rotation rotation0 = Rotation.IDENTITY;
      double[] doubleArray0 = rotation0.getAngles(rotationOrder0);
      assertArrayEquals(new double[] {0.0, -0.0, 0.0}, doubleArray0, 0.01);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Rotation rotation0 = new Rotation(0.027777772389658586, 0.027777772389658586, 0.027777772389658586, 0.027777772389658586, true);
      RotationOrder rotationOrder0 = RotationOrder.YXZ;
      try { 
        rotation0.getAngles(rotationOrder0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Cardan angles singularity
         //
         verifyException("org.apache.commons.math.geometry.euclidean.threed.Rotation", e);
      }
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      Rotation rotation0 = new Rotation((-1838.652), (-0.0012021211041601761), (-1838.652), 13.07, false);
      RotationOrder rotationOrder0 = RotationOrder.YXZ;
      try { 
        rotation0.getAngles(rotationOrder0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Cardan angles singularity
         //
         verifyException("org.apache.commons.math.geometry.euclidean.threed.Rotation", e);
      }
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      RotationOrder rotationOrder0 = RotationOrder.YXZ;
      Rotation rotation0 = Rotation.IDENTITY;
      double[] doubleArray0 = rotation0.getAngles(rotationOrder0);
      assertArrayEquals(new double[] {0.0, -0.0, 0.0}, doubleArray0, 0.01);
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      Rotation rotation0 = new Rotation(1.6668541182173628, 1.6668541182173628, (-1314.2506), 1.6668541182173628, true);
      double[] doubleArray0 = rotation0.getAngles((RotationOrder) null);
      assertArrayEquals(new double[] {(-0.784129871623817), 3.1380053875437692, (-0.7866664551710797)}, doubleArray0, 0.01);
      assertEquals(0.0012682893934880125, rotation0.getQ0(), 0.01);
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      Rotation rotation0 = new Rotation((-638.06220555), (-638.06220555), 2.0, (-638.06220555), false);
      RotationOrder rotationOrder0 = RotationOrder.YZX;
      try { 
        rotation0.getAngles(rotationOrder0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Cardan angles singularity
         //
         verifyException("org.apache.commons.math.geometry.euclidean.threed.Rotation", e);
      }
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      RotationOrder rotationOrder0 = RotationOrder.YZX;
      Rotation rotation0 = new Rotation(19.604189974750113, (-638.06220555), (-638.06220555), 19.604189974750113, false);
      try { 
        rotation0.getAngles(rotationOrder0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Cardan angles singularity
         //
         verifyException("org.apache.commons.math.geometry.euclidean.threed.Rotation", e);
      }
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      RotationOrder rotationOrder0 = RotationOrder.ZXY;
      Rotation rotation0 = Rotation.IDENTITY;
      double[] doubleArray0 = rotation0.getAngles(rotationOrder0);
      assertArrayEquals(new double[] {-0.0, 0.0, -0.0}, doubleArray0, 0.01);
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      RotationOrder rotationOrder0 = RotationOrder.ZXY;
      Rotation rotation0 = new Rotation(0.5, 0.5, 0.5, (-0.5032059823641271), false);
      try { 
        rotation0.getAngles(rotationOrder0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Cardan angles singularity
         //
         verifyException("org.apache.commons.math.geometry.euclidean.threed.Rotation", e);
      }
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      RotationOrder rotationOrder0 = RotationOrder.ZXY;
      Rotation rotation0 = new Rotation(0.5, 0.5, 0.5, 33.91727005985855, false);
      try { 
        rotation0.getAngles(rotationOrder0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Cardan angles singularity
         //
         verifyException("org.apache.commons.math.geometry.euclidean.threed.Rotation", e);
      }
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      Vector3D vector3D0 = Vector3D.MINUS_K;
      RotationOrder rotationOrder0 = RotationOrder.ZYX;
      Rotation rotation0 = new Rotation(vector3D0, vector3D0, vector3D0, vector3D0);
      double[] doubleArray0 = rotation0.getAngles(rotationOrder0);
      assertArrayEquals(new double[] {Double.NaN, Double.NaN, Double.NaN}, doubleArray0, 0.01);
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      Rotation rotation0 = new Rotation(1.6077569201477233, (-495.70386), 1.6077569201477233, 7678768.572387066, false);
      RotationOrder rotationOrder0 = RotationOrder.ZYX;
      try { 
        rotation0.getAngles(rotationOrder0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Cardan angles singularity
         //
         verifyException("org.apache.commons.math.geometry.euclidean.threed.Rotation", e);
      }
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      Rotation rotation0 = new Rotation(0.027777772389658586, 0.027777772389658586, 0.027777772389658586, 0.027777772389658586, true);
      RotationOrder rotationOrder0 = RotationOrder.ZYX;
      try { 
        rotation0.getAngles(rotationOrder0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Cardan angles singularity
         //
         verifyException("org.apache.commons.math.geometry.euclidean.threed.Rotation", e);
      }
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      RotationOrder rotationOrder0 = RotationOrder.XYX;
      Vector3D vector3D0 = rotationOrder0.getA2();
      Rotation rotation0 = new Rotation(vector3D0, vector3D0, vector3D0, vector3D0);
      double[] doubleArray0 = rotation0.getAngles(rotationOrder0);
      assertEquals(3, doubleArray0.length);
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      RotationOrder rotationOrder0 = RotationOrder.XYX;
      Vector3D vector3D0 = Vector3D.PLUS_K;
      Vector3D vector3D1 = new Vector3D(6.889733407866442E-5, vector3D0, (-234.28736102158425), vector3D0);
      Rotation rotation0 = new Rotation(vector3D0, vector3D1);
      try { 
        rotation0.getAngles(rotationOrder0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Euler angles singularity
         //
         verifyException("org.apache.commons.math.geometry.euclidean.threed.Rotation", e);
      }
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      RotationOrder rotationOrder0 = RotationOrder.XYX;
      Rotation rotation0 = Rotation.IDENTITY;
      try { 
        rotation0.getAngles(rotationOrder0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Euler angles singularity
         //
         verifyException("org.apache.commons.math.geometry.euclidean.threed.Rotation", e);
      }
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      Vector3D vector3D0 = Vector3D.NaN;
      Rotation rotation0 = new Rotation(vector3D0, vector3D0, vector3D0, vector3D0);
      RotationOrder rotationOrder0 = RotationOrder.XZX;
      double[] doubleArray0 = rotation0.getAngles(rotationOrder0);
      assertEquals(3, doubleArray0.length);
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      Rotation rotation0 = new Rotation(0.0, 0.0, 774.4080454125, 774.4080454125, false);
      RotationOrder rotationOrder0 = RotationOrder.XZX;
      try { 
        rotation0.getAngles(rotationOrder0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Euler angles singularity
         //
         verifyException("org.apache.commons.math.geometry.euclidean.threed.Rotation", e);
      }
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      Vector3D vector3D0 = Vector3D.PLUS_K;
      Rotation rotation0 = new Rotation(vector3D0, vector3D0);
      RotationOrder rotationOrder0 = RotationOrder.XZX;
      try { 
        rotation0.getAngles(rotationOrder0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Euler angles singularity
         //
         verifyException("org.apache.commons.math.geometry.euclidean.threed.Rotation", e);
      }
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      Vector3D vector3D0 = new Vector3D((-0.33333333333333287), 0.7385090212904348, (-0.33333333333333287));
      Rotation rotation0 = new Rotation(vector3D0, vector3D0, vector3D0, vector3D0);
      RotationOrder rotationOrder0 = RotationOrder.YXY;
      double[] doubleArray0 = rotation0.getAngles(rotationOrder0);
      assertEquals(3, doubleArray0.length);
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      Rotation rotation0 = new Rotation((-2.9156920635172886E-13), 1.0, 1.2958646899018938E-9, (-2.9156920635172886E-13), false);
      RotationOrder rotationOrder0 = RotationOrder.YXY;
      try { 
        rotation0.getAngles(rotationOrder0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Euler angles singularity
         //
         verifyException("org.apache.commons.math.geometry.euclidean.threed.Rotation", e);
      }
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      RotationOrder rotationOrder0 = RotationOrder.YXY;
      Rotation rotation0 = Rotation.IDENTITY;
      try { 
        rotation0.getAngles(rotationOrder0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Euler angles singularity
         //
         verifyException("org.apache.commons.math.geometry.euclidean.threed.Rotation", e);
      }
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      Vector3D vector3D0 = Vector3D.MINUS_I;
      Rotation rotation0 = new Rotation(vector3D0, vector3D0, vector3D0, vector3D0);
      RotationOrder rotationOrder0 = RotationOrder.YZY;
      double[] doubleArray0 = rotation0.getAngles(rotationOrder0);
      assertEquals(3, doubleArray0.length);
  }

  @Test(timeout = 4000)
  public void test55()  throws Throwable  {
      Rotation rotation0 = new Rotation(0.0, 0.0, 0.0, 3941.79372073, false);
      RotationOrder rotationOrder0 = RotationOrder.YZY;
      try { 
        rotation0.getAngles(rotationOrder0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Euler angles singularity
         //
         verifyException("org.apache.commons.math.geometry.euclidean.threed.Rotation", e);
      }
  }

  @Test(timeout = 4000)
  public void test56()  throws Throwable  {
      RotationOrder rotationOrder0 = RotationOrder.YZY;
      Rotation rotation0 = Rotation.IDENTITY;
      try { 
        rotation0.getAngles(rotationOrder0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Euler angles singularity
         //
         verifyException("org.apache.commons.math.geometry.euclidean.threed.Rotation", e);
      }
  }

  @Test(timeout = 4000)
  public void test57()  throws Throwable  {
      Vector3D vector3D0 = Vector3D.MINUS_K;
      Rotation rotation0 = new Rotation(vector3D0, vector3D0, vector3D0, vector3D0);
      RotationOrder rotationOrder0 = RotationOrder.ZXZ;
      double[] doubleArray0 = rotation0.getAngles(rotationOrder0);
      assertEquals(3, doubleArray0.length);
  }

  @Test(timeout = 4000)
  public void test58()  throws Throwable  {
      Rotation rotation0 = Rotation.IDENTITY;
      RotationOrder rotationOrder0 = RotationOrder.ZXZ;
      try { 
        rotation0.getAngles(rotationOrder0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Euler angles singularity
         //
         verifyException("org.apache.commons.math.geometry.euclidean.threed.Rotation", e);
      }
  }

  @Test(timeout = 4000)
  public void test59()  throws Throwable  {
      RotationOrder rotationOrder0 = RotationOrder.ZYZ;
      Vector3D vector3D0 = rotationOrder0.getA1();
      Vector3D vector3D1 = new Vector3D((-1019.2), vector3D0, (-1019.2), vector3D0);
      Rotation rotation0 = new Rotation(vector3D0, vector3D1);
      try { 
        rotation0.getAngles(rotationOrder0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Euler angles singularity
         //
         verifyException("org.apache.commons.math.geometry.euclidean.threed.Rotation", e);
      }
  }

  @Test(timeout = 4000)
  public void test60()  throws Throwable  {
      RotationOrder rotationOrder0 = RotationOrder.ZYZ;
      Rotation rotation0 = Rotation.IDENTITY;
      try { 
        rotation0.getAngles(rotationOrder0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Euler angles singularity
         //
         verifyException("org.apache.commons.math.geometry.euclidean.threed.Rotation", e);
      }
  }

  @Test(timeout = 4000)
  public void test61()  throws Throwable  {
      double[][] doubleArray0 = new double[3][3];
      Rotation rotation0 = null;
      try {
        rotation0 = new Rotation(doubleArray0, (-2828.5643));
        fail("Expecting exception: Exception");
      
      } catch(Throwable e) {
         //
         // unable to orthogonalize matrix in 10 iterations
         //
         verifyException("org.apache.commons.math.geometry.euclidean.threed.Rotation", e);
      }
  }
}
